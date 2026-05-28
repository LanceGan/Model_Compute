# 模型对算力等效建模评估工具 — 设计文档

## 1. 项目概述

### 1.1 目标

构建一个统一的算力等效建模评估工具，输入模型参数（负载类型、参数量、量化方案、并发量、tokens数），输出所需的硬件配置（硬件型号、单卡显存、卡数、内存、带宽）和预估性能。

### 1.2 评分权重

- 模型覆盖度：40%（支持多种模型类型、参数、量化方案、硬件资源）
- 度量算法精度：40%（预测值与实际运行真值的偏差）
- 代码文档完整性：20%（模块化设计、注释、文档）

### 1.3 支持的模型类型

| 类型 | 典型代表 | 关键特征 |
|------|----------|----------|
| 稠密模型 (Dense) | LLaMA, GPT, Qwen | 全部参数参与计算 |
| MoE 模型 | Mixtral, DeepSeek-V2 | 稀疏激活，只激活部分专家 |
| 类o1推理模型 | DeepSeek-R1 | 推理时生成大量中间思考token |
| 多模态模型 | LLaVA, Qwen-VL | 视觉编码器 + 语言模型 |

### 1.4 支持的硬件

- **NVIDIA GPU**：A100 (40G/80G)、H100 (80G)、H200、L40S、RTX 4090
- **昇腾 NPU**：910A、910B、910C、300I Duo、310P
- **其他国产**：寒武纪 MLU370、海光 DCU、沐曦 N100
- **扩展**：支持用户自定义添加新硬件

## 2. 系统架构

### 2.1 三层架构

```
┌─────────────────────────────────────────────┐
│         展示层 (Python + Streamlit)          │
│   Web UI、表单输入、结果可视化、报告导出      │
├─────────────────────────────────────────────┤
│              ↕ pybind11 绑定                 │
├─────────────────────────────────────────────┤
│            业务逻辑层 (Python)               │
│   模型分析器、硬件数据库、校准数据管理        │
├─────────────────────────────────────────────┤
│              ↕ 调用                          │
├─────────────────────────────────────────────┤
│           计算引擎层 (C++)                   │
│   FLOPs计算、显存估算、带宽分析、硬件匹配    │
│   并行策略优化、校准系数拟合                 │
└─────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 展示层 | Streamlit | Web UI、交互式表单、图表 |
| 业务逻辑层 | Python 3.10+ | 模型分析、数据管理、参数验证 |
| 计算引擎层 | C++17 | 核心计算、优化搜索 |
| 桥接层 | pybind11 | C++/Python 绑定 |
| 构建 | CMake | C++ 编译 |
| 数据存储 | JSON | 硬件规格、模型预设、校准数据 |
| 测试 | Google Test (C++) + pytest (Python) | 单元测试 |

## 3. 核心估算引擎（C++）

### 3.1 Dense 模型估算

**显存估算：**

```
模型权重 = 参数量 × 每参数字节数
  FP16: 2 bytes/param, INT8: 1 byte/param, INT4: 0.5 bytes/param

KV Cache = 2 × num_layers × hidden_dim × seq_len × batch_size × 每参数字节数
  其中 2 代表 Key 和 Value 两个缓存

激活值 ≈ 参数量 × activation_ratio (典型值 0.01~0.05，与 batch_size 相关)

总显存 = (模型权重 + KV Cache + 激活值) × (1 + overhead_ratio)
  overhead_ratio ≈ 0.10 (运行时开销)
```

**算力估算 (FLOPs)：**

```
Prefill 阶段 = 2 × 参数量 × 输入序列长度
  （矩阵乘法：每个参数参与 2 次浮点运算）

Decode 阶段（每 token） ≈ 2 × 参数量

总 FLOPs = Prefill FLOPs + Decode FLOPs × 输出 token 数
```

**吞吐量瓶颈分析：**

```
计算受限吞吐:
  tokens/s = 算力(TFLOPS) × 10^12 / (2 × 参数量)

带宽受限吞吐:
  tokens/s = 带宽(GB/s) × 10^9 / (参数量 × 每参数字节数 + KV_per_token)

实际吞吐 = min(计算受限, 带宽受限) × 并行效率系数
  并行效率系数：单卡 0.85~0.95，多卡 0.70~0.85
```

**依据：** 基于 Transformer 架构的计算特性。Prefill 阶段是计算密集型（compute-bound），Decode 阶段是访存密集型（memory-bound）。这一分析框架来源于 NVIDIA 的 Megatron-LM 分析方法和 arXiv:1909.08053 (Alpa)。

### 3.2 MoE 模型估算

MoE 模型的关键区别在于**总参数量 vs 激活参数量**：

```
总参数量 = 所有专家权重之和 + 共享层参数（Attention + Router）
激活参数量 = 每token激活的专家数 × 单专家参数 + 共享层参数

显存 = 总参数量 × 每参数字节数 + KV Cache + 激活值 + 开销
  （所有专家权重需常驻显存）

FLOPs = 激活参数量 × 2 × tokens + 路由开销
  路由开销 ≈ 总 FLOPs × 0.01

吞吐量瓶颈分析：
  计算受限: tokens/s = 算力 × 10^12 / (2 × 激活参数量)
  带宽受限: tokens/s = 带宽 × 10^9 / (总参数量 × 每参数字节数 / TP + KV_per_token)
```

**依据：** MoE 架构的稀疏激活特性（Fedus et al., 2022; DeepSeek-V2 技术报告）。显存需存储全部专家，但计算只激活部分专家，这是 MoE 的核心优势。

### 3.3 类o1 推理模型估算

类o1模型的核心特征是**中间推理token**：

```
实际输出 token 数 = 用户可见输出 + 内部推理 token

推理 token 倍率：
  轻度推理: 2x~3x（简单问答）
  中度推理: 3x~5x（数学/代码）
  重度推理: 5x~10x（复杂推理）

总计算量 = Dense 基础公式 × (1 + 推理token倍率)

KV Cache 增长：
  推理阶段 KV Cache = 2 × layers × hidden × (input_seq + reasoning_tokens) × batch × bytes
  推理token会显著增加 KV Cache 占用
```

**依据：** 参考 OpenAI o1 和 DeepSeek-R1 的推理增强机制。推理模型通过生成 chain-of-thought token 来提升准确率，但代价是显著增加计算量和显存占用。

### 3.4 多模态模型估算

多模态模型由视觉编码器 + 语言模型骨干组成：

```
组成部分：
  视觉编码器 (ViT)：独立的显存和算力需求
  语言模型骨干：复用 Dense/MoE 估算逻辑
  模态融合层（投影层）：额外显存开销

显存 = 语言模型显存 + 视觉编码器显存 + 图像token的KV Cache
  图像token数 = (图像分辨率 / patch_size)^2
  典型值：224px→196 tokens, 336px→576 tokens, 448px→1024 tokens

FLOPs = 语言模型FLOPs + 视觉编码器FLOPs × 图像数量
  ViT FLOPs ≈ 2 × ViT参数量 × 图像token数

带宽需求额外考虑视觉编码器的权重加载
```

**依据：** 参考 LLaVA (Liu et al., 2023) 和 Qwen-VL (Bai et al., 2023) 的架构设计。视觉编码器通常基于 ViT-L/14 (304M params) 或 InternViT (6B params)。

## 4. 硬件数据库

### 4.1 数据结构

每条硬件记录包含以下字段：

```json
{
  "name": "NVIDIA A100 80GB",
  "vendor": "NVIDIA",
  "architecture": "Ampere",
  "type": "GPU",
  "specs": {
    "fp16_tflops": 312,
    "int8_tops": 624,
    "fp32_tflops": 19.5,
    "memory_gb": 80,
    "memory_type": "HBM2e",
    "memory_bandwidth_gbs": 2039,
    "nvlink_bandwidth_gbs": 600,
    "pcie_version": "4.0",
    "max_tdp_watts": 300
  },
  "notes": "主流训练/推理GPU，支持NVLink多卡互联"
}
```

### 4.2 预置硬件库

| 厂商 | 型号 | FP16 TFLOPS | 显存 | 带宽 |
|------|------|-------------|------|------|
| NVIDIA | A100 40G | 312 | 40GB HBM2e | 1555 GB/s |
| NVIDIA | A100 80G | 312 | 80GB HBM2e | 2039 GB/s |
| NVIDIA | H100 80G | 990 | 80GB HBM3 | 3350 GB/s |
| NVIDIA | H200 | 990 | 141GB HBM3e | 4800 GB/s |
| NVIDIA | L40S | 362 | 48GB GDDR6X | 864 GB/s |
| NVIDIA | RTX 4090 | 330 | 24GB GDDR6X | 1008 GB/s |
| 昇腾 | 910B | 320 | 64GB HBM2e | 1200 GB/s |
| 昇腾 | 300I Duo | 256 | 64GB (2×32) | 800 GB/s |
| 寒武纪 | MLU370 | 256 | 48GB | 614 GB/s |

> 注：以上数据为参考值，实际以厂商官方规格为准。工具支持用户自行修改。

### 4.3 硬件匹配算法

```
输入：模型估算结果（显存需求、算力需求、带宽需求）、目标吞吐量
输出：推荐硬件配置列表（按性价比排序）

算法：
1. 对每种硬件 h in hardware_pool:
   a. 计算单卡可承载模型大小 → 是否需要模型并行
   b. 计算所需卡数:
      cards_by_memory = ceil(显存需求 / h.memory_gb)
      cards_by_compute = ceil(算力需求 / h.fp16_tflops)
      cards = max(cards_by_memory, cards_by_compute)
   c. 选择并行策略:
      if cards <= 8: 优先张量并行(TP)
      if cards > 8: TP + 流水线并行(PP) 组合
      if MoE: 加入专家并行(EP)
   d. 估算通信开销:
      TP通信 = 2 × hidden_dim × bytes × (TP-1) / TP / nvlink_bandwidth
      PP通信 = activations × (PP-1) / PP / bandwidth
   e. 重新计算实际吞吐（含通信开销）
   f. 验证是否满足基线吞吐 ≥ 10 tokens/s

2. 按 性价比 = 预估吞吐 / (卡数 × 单卡成本) 排序
3. 输出 Top-N 推荐
```

## 5. 校准模块

### 5.1 校准机制

目的：用少量实测数据修正理论公式的系统性误差。

```
校准系数定义：
  throughput_factor = 实际吞吐 / 理论吞吐  (典型值: 0.6 ~ 0.9)
  memory_factor = 实际显存 / 理论显存      (典型值: 1.05 ~ 1.2)

校准流程：
  1. 在可用GPU上运行代表性模型配置
  2. 记录实际吞吐量、显存占用
  3. 与理论预测值对比
  4. 按 (模型类型, 硬件类型) 分组计算校准系数
  5. 持久化存储校准系数

预测时：
  adjusted_throughput = theoretical_throughput × throughput_factor
  adjusted_memory = theoretical_memory × memory_factor

未校准的配置使用默认系数 1.0，不引入额外误差。
```

### 5.2 校准数据管理

- 支持批量导入实测数据（CSV格式）
- 支持手动输入单条校准记录
- 校准系数自动按模型类型×硬件类型分组
- 支持查看和编辑已有校准系数

## 6. Web UI 设计

### 6.1 页面结构

**页面1：算力估算（主页面）**

左侧输入面板：
- 负载类型选择（稠密/MoE/类o1/多模态）
- 模型参数量（预设 + 自定义输入）
- 量化方案（FP16/INT8/INT4）
- 并发量（1~500）
- 最大tokens数（512~32768）
- 条件参数：类o1的推理深度、多模态的图像分辨率

右侧结果展示：
- 推荐硬件配置表（型号、卡数、显存、内存、带宽）
- 性能预估（吞吐、延迟、瓶颈类型）
- 满足/不满足基线吞吐的标识

**页面2：多硬件对比**
- 表格展示所有可选硬件方案
- 柱状图对比吞吐量和成本
- 推荐最优方案高亮

**页面3：敏感性分析**
- 滑动条调整并发量/序列长度
- 实时曲线图展示吞吐/卡数变化
- 识别性能拐点

**页面4：管理**
- 硬件数据库管理（增删改查）
- 校准数据管理（导入/查看/编辑）
- 模型预设管理

## 7. 项目目录结构

```
model-compute/
├── cpp/                          # C++ 核心引擎
│   ├── include/
│   │   ├── estimation_engine.h
│   │   ├── hardware_matcher.h
│   │   └── calibration.h
│   ├── src/
│   │   ├── estimation_engine.cpp
│   │   ├── hardware_matcher.cpp
│   │   └── calibration.cpp
│   ├── bindings/
│   │   └── pybind_module.cpp     # pybind11 绑定
│   ├── tests/
│   │   └── test_engine.cpp
│   ├── CMakeLists.txt
│   └── README.md
├── python/                       # Python 业务层
│   ├── core/
│   │   ├── model_analyzer.py     # 模型架构分析
│   │   ├── hardware_db.py        # 硬件数据库管理
│   │   └── calibration_mgr.py    # 校准数据管理
│   ├── data/
│   │   ├── hardware_specs.json   # 硬件规格数据
│   │   ├── model_presets.json    # 预置模型配置
│   │   └── calibration_data/     # 校准数据
│   ├── web/
│   │   ├── app.py                # Streamlit 主入口
│   │   ├── pages/
│   │   │   ├── estimation.py     # 估算页面
│   │   │   ├── comparison.py     # 对比页面
│   │   │   ├── sensitivity.py    # 敏感性分析
│   │   │   └── management.py     # 硬件/校准管理
│   │   └── components/
│   │       └── charts.py         # 图表组件
│   ├── tests/
│   │   └── test_*.py
│   ├── setup.py
│   └── README.md
├── docs/                         # 文档
│   ├── architecture.md
│   ├── formulas.md               # 公式推导和参考文献
│   ├── usage.md                  # 使用说明
│   └── calibration_guide.md      # 校准指南
├── scripts/
│   ├── build.sh                  # 编译C++模块
│   ├── run_calibration.py        # 校准脚本
│   └── run_tests.sh              # 运行全部测试
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

## 8. 参考文献

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", arXiv:1909.08053
- Zheng et al., "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning", OSDI 2022
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", JMLR 2022
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434
- Liu et al., "Visual Instruction Tuning", NeurIPS 2023 (LLaVA)
- Bai et al., "Qwen-VL: A Versatile Vision-Language Model", arXiv:2308.12966
- NVIDIA, "A100 Tensor Core GPU Architecture" (技术白皮书)
- 华为, "Ascend 910B 技术规格" (官方文档)
