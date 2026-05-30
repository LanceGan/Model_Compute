# 模型对算力等效建模评估工具

基于异构算力资源池，支持多负载的算力需求表征框架和动态调度工具。输入模型参数，输出所需硬件配置和预估性能。

## 功能亮点

- **5 种模型类型**：稠密 (Dense)、MoE、类 o1 推理、多模态、生成式推荐
- **14 种硬件**：NVIDIA GPU (A100/H100/H200/L40S/RTX4090) + 昇腾 NPU (910A/910B/910C/300I Duo/310P) + 寒武纪 MLU370 + 海光 DCU + 沐曦 N100
- **23 个模型预设**：LLaMA-2/3、Qwen-2.5、Mixtral、DeepSeek-V2/V3/R1、LLaVA、DLRM、SASRec 等
- **高精度估算**：MAPE 4.9%（未校准），支持校准数据自动修正
- **GQA/SwiGLU 支持**：精确建模 Grouped Query Attention 和 SwiGLU FFN 架构
- **动态调度**：批量估算、CSV 导入导出、多硬件对比

## 支持的模型类型

| 类型 | 典型代表 | 关键特征 |
|------|----------|----------|
| 稠密模型 (Dense) | LLaMA-2/3, Qwen-2.5, GPT | 全部参数参与计算 |
| MoE 模型 | Mixtral, DeepSeek-V2/V3 | 稀疏激活，只激活部分专家 |
| 类o1推理模型 | DeepSeek-R1, QwQ | 推理时生成大量中间思考 token |
| 多模态模型 | LLaVA, Qwen-VL | 视觉编码器 + 语言模型 |
| 生成式推荐模型 | DLRM, DeepFM, SASRec | Embedding Table + MLP/Transformer |

## 支持的硬件

| 厂商 | 型号 | FP16 TFLOPS | 显存 | 带宽 |
|------|------|-------------|------|------|
| NVIDIA | A100 40GB | 312 | 40GB HBM2e | 1555 GB/s |
| NVIDIA | A100 80GB | 312 | 80GB HBM2e | 2039 GB/s |
| NVIDIA | H100 80GB | 990 | 80GB HBM3 | 3350 GB/s |
| NVIDIA | H200 | 990 | 141GB HBM3e | 4800 GB/s |
| NVIDIA | L40S | 362 | 48GB GDDR6X | 864 GB/s |
| NVIDIA | RTX 4090 | 330 | 24GB GDDR6X | 1008 GB/s |
| 昇腾 | 910A | 320 | 32GB HBM2 | 1200 GB/s |
| 昇腾 | 910B | 320 | 64GB HBM2e | 1200 GB/s |
| 昇腾 | 910C | 400 | 64GB HBM2e | 1600 GB/s |
| 昇腾 | 300I Duo | 256 | 64GB HBM2e | 800 GB/s |
| 昇腾 | 310P | 150 | 24GB LPDDR4x | 100 GB/s |
| 寒武纪 | MLU370 | 256 | 48GB HBM2e | 614 GB/s |
| 海光 | DCU | 200 | 64GB HBM2 | 800 GB/s |
| 沐曦 | N100 | 200 | 48GB HBM2e | 600 GB/s |

## 快速开始

### 安装

```bash
# 1. 编译 C++ 模块
# Windows:
scripts\build.bat
# Linux/macOS:
bash scripts/build.sh

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装项目
pip install -e .
```

### 启动 Web UI

```bash
streamlit run python/web/app.py
```

浏览器打开 http://localhost:8501

### 运行测试

```bash
# C++ 测试
cd build && ctest --output-on-failure

# Python 测试
python -m pytest python/tests/ -v

# 精度验证
python scripts/validate_accuracy.py
```

## 架构概览

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

## 精度验证

| 模型类型 | 预测误差 (MAPE) |
|---------|----------------|
| Dense (稠密) | 4.8% |
| MoE | 5.1-12.2% |
| o1 推理 | 0.5-7.7% |
| 多模态 | 1.9-4.6% |
| 推荐模型 | 8.3% |
| **整体** | **4.9%** |

校准数据支持自动修正，含校准后预计误差 < 3%。

## Python API

```python
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager
import model_compute as mc

# 创建模型参数
analyzer = ModelAnalyzer()
params = analyzer.create_params(
    model_type="dense",
    preset_name="LLaMA-3 8B",
    quant="FP16",
    concurrency=1,
    max_tokens=2048,
)

# 估算
engine = mc.EstimationEngine()
result = engine.estimate(params)
print(f"显存需求: {result.memory_gb:.1f} GB")
print(f"FLOPs: {result.flops_total:.2e}")
print(f"运行时开销: {result.runtime_overhead_gb:.1f} GB")

# 硬件匹配
hw_db = HardwareDB()
matcher = mc.HardwareMatcher()
hw_pool = hw_db.to_cpp_hardware_list()
configs = matcher.match(result, params, hw_pool, baseline_throughput=10.0)

for c in configs[:3]:
    print(f"{c.hardware.name}: {c.num_cards}卡, {c.estimated_throughput:.0f} tok/s")
```

## 项目结构

```
model-compute/
├── cpp/                          # C++ 核心引擎
│   ├── include/                  # 头文件
│   │   ├── estimation_engine.h   # 估算引擎接口
│   │   ├── hardware_matcher.h    # 硬件匹配器接口
│   │   ├── calibration.h         # 校准模块接口
│   │   └── architecture_utils.h  # 共享架构推断
│   ├── src/                      # 实现
│   ├── bindings/pybind_module.cpp # Python 绑定
│   └── tests/                    # Google Test 测试
├── python/                       # Python 业务层
│   ├── core/                     # 核心业务逻辑
│   ├── data/                     # 数据文件 (JSON, CSV)
│   ├── web/                      # Streamlit Web UI
│   └── tests/                    # pytest 测试
├── docs/                         # 文档
├── scripts/                      # 构建和验证脚本
└── README.md
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 展示层 | Streamlit | Web UI、交互式表单、图表 |
| 业务逻辑层 | Python 3.10+ | 模型分析、数据管理、参数验证 |
| 计算引擎层 | C++17 | 核心计算、优化搜索 |
| 桥接层 | pybind11 | C++/Python 绑定 |
| 构建 | CMake | C++ 编译 |
| 数据存储 | JSON/CSV | 硬件规格、模型预设、校准数据 |
| 测试 | Google Test + pytest | 单元测试 + 集成测试 |

## 参考文献

- Shoeybi et al., "Megatron-LM", arXiv:1909.08053
- Fedus et al., "Switch Transformers", JMLR 2022
- DeepSeek-AI, "DeepSeek-V2", arXiv:2405.04434
- Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023
- Naumov et al., "DLRM", arXiv:1906.00091
- NVIDIA, "A100 Tensor Core GPU Architecture"
- 华为, "Ascend 910B 技术规格"
