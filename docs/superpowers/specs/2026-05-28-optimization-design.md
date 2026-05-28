# 算力等效建模评估工具 — 优化设计文档

## 1. 优化目标

基于赛题评分标准，针对当前实现的差距进行优化：

- **模型覆盖度（40%）**：补充生成式推荐模型支持，扩充硬件数据库，完善建模文档
- **度量算法精度（40%）**：改进理论公式 + 预置公开基准校准系数
- **代码文档完整性（20%）**：增强公式推导文档、代码注释

## 2. 优化范围

### 2.1 生成式推荐模型支持（新功能）

**目标：** 支持 DLRM/DeepFM 和序列推荐 Transformer 两类推荐模型。

**数据结构扩展：**

在 `ModelParams` 中新增推荐模型专用字段：

```cpp
int num_sparse_features = 0;      // 稀疏特征数（典型值: 10~100）
int vocab_size_per_feature = 0;   // 每个特征的词表大小（典型值: 10K~10M）
int embed_dim = 0;                // Embedding 维度（典型值: 64/128/256）
int num_mlp_layers = 0;           // MLP 层数
std::vector<int> mlp_dims;        // MLP 每层维度（如 [1024, 512, 256]）
```

**DLRM/DeepFM 估算逻辑：**

推荐模型的核心特征是 Embedding Table 占用大量显存，与 Dense 模型完全不同。

```
Embedding 显存 = num_sparse_features × vocab_size_per_feature × embed_dim × bytes_per_param(quant)

MLP 参数量 = Σ(mlp_dims[i] × mlp_dims[i+1]) + Σ(mlp_dims[i])
MLP 显存 = MLP 参数量 × bytes_per_param(quant)

总显存 = (Embedding 显存 + MLP 显存) × (1 + overhead_ratio)
  overhead_ratio ≈ 0.10

计算量（FLOPs）：
  Embedding 查找 = 访存操作，不计入 FLOPs
  MLP FLOPs = 2 × Σ(mlp_dims[i] × mlp_dims[i+1]) × concurrency

带宽需求：
  Embedding 查找是随机访存，带宽受限
  bandwidth_gbs = num_active_lookups × embed_dim × bytes × target_throughput / 1e9
```

**序列推荐 Transformer 估算逻辑：**

```
复用 Dense 估算 + Item Embedding 额外显存
  （复用 vocab_size_per_feature 作为 item 词表大小，无需新增字段）
Item Embedding 显存 = vocab_size_per_feature × embed_dim × bytes_per_param
计算量 = 同 Transformer（Prefill + Decode）
```

**预设模型：**

```json
{
  "recommendation": [
    {
      "name": "DLRM-small",
      "num_sparse_features": 26,
      "vocab_size_per_feature": 100000,
      "embed_dim": 128,
      "mlp_dims": [512, 256, 1],
      "notes": "Facebook DLRM 基准配置"
    },
    {
      "name": "DLRM-large",
      "num_sparse_features": 26,
      "vocab_size_per_feature": 1000000,
      "embed_dim": 128,
      "mlp_dims": [1024, 512, 256, 1],
      "notes": "大规模推荐模型"
    },
    {
      "name": "DeepFM",
      "num_sparse_features": 39,
      "vocab_size_per_feature": 500000,
      "embed_dim": 64,
      "mlp_dims": [1024, 512, 1],
      "notes": "DeepFM 基准配置"
    },
    {
      "name": "SASRec",
      "num_sparse_features": 1,
      "vocab_size_per_feature": 50000,
      "embed_dim": 64,
      "notes": "序列推荐 Transformer"
    }
  ]
}
```

**涉及文件：**
- `cpp/include/estimation_engine.h` — 新增 `ModelType::RECOMMENDATION` 和相关字段
- `cpp/src/estimation_engine.cpp` — 实现 `estimate_recommendation()`
- `python/data/model_presets.json` — 添加推荐模型预设
- `python/core/model_analyzer.py` — 支持推荐模型参数创建
- `python/web/pages/estimation.py` — UI 支持推荐模型条件参数
- `python/web/pages/comparison.py` — 同上
- `python/web/pages/sensitivity.py` — 同上
- `cpp/tests/test_engine.cpp` — 推荐模型估算测试
- `python/tests/test_integration.py` — 推荐模型集成测试

### 2.2 估算精度提升（B2 方案）

**目标：** 改进理论公式 + 预置公开基准校准系数，提升度量算法精度。

#### 2.2.1 架构推断改进

当前用 LLaMA 系列的固定映射表。改进为参数化公式：

```cpp
static void infer_architecture(double param_b, int& num_layers, int& hidden_dim) {
    // 依据：Transformer 参数量 ≈ 12 × L × d² (忽略 embedding)
    // 典型 ratio: d/L ≈ 128 (LLaMA 系列)
    double p = param_b * 1e9;
    double L = std::pow(p / (12.0 * 128.0 * 128.0), 1.0/3.0);
    num_layers = std::max(2, static_cast<int>(std::round(L)));
    hidden_dim = std::max(512, static_cast<int>(std::round(128.0 * num_layers)));
    // 对齐到 64 的倍数（硬件友好的对齐方式）
    hidden_dim = (hidden_dim + 63) / 64 * 64;
}
```

#### 2.2.2 激活值估算改进

当前用固定 ratio，改进为基于 batch_size 和 seq_len 的公式：

```
激活值 ≈ batch_size × seq_len × hidden_dim × num_layers × bytes_per_param × activation_factor
  推理模式 activation_factor ≈ 2（不需要保存反向传播中间结果）
  训练模式 activation_factor ≈ 16
```

#### 2.2.3 通信开销模型改进

当前用简化比例公式，改进为区分互联类型：

```cpp
double estimate_comm_overhead(int cards, const HardwareSpec& hw, const ModelParams& mp) {
    if (cards <= 1) return 0.0;

    int num_layers, hidden_dim;
    infer_architecture(mp.param_billions, num_layers, hidden_dim);
    double bpp = bytes_per_param(mp.quant);

    // AllReduce 通信量: 2 × hidden_dim × bytes × (N-1)/N per layer
    double comm_bytes_per_layer = 2.0 * hidden_dim * bpp * (cards - 1) / cards;
    double total_comm_bytes = comm_bytes_per_layer * num_layers;

    // 选择互联带宽
    double bandwidth_gbs = 0;
    if (hw.nvlink_bandwidth_gbs > 0) {
        bandwidth_gbs = hw.nvlink_bandwidth_gbs;
    } else if (hw.vendor == "华为") {
        bandwidth_gbs = 200.0;  // HCCS 典型带宽
    } else {
        bandwidth_gbs = 64.0;   // PCIe 4.0 x16
    }

    // 通信时间 vs 计算时间
    double comm_time = total_comm_bytes / (bandwidth_gbs * 1e9);
    double compute_time = 2.0 * mp.param_billions * 1e9 / (hw.fp16_tflops * 1e12);
    return comm_time / (comm_time + compute_time);
}
```

#### 2.2.4 基准校准数据预置

收集公开基准数据，预置默认校准系数。在 `python/data/calibration_data/default_calibration.csv` 中存储：

```csv
# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem,source
dense,NVIDIA A100 80GB,20.0,15.0,14.0,15.5,MLPerf Inference v3.1
dense,NVIDIA H100 80GB,60.0,48.0,14.0,15.2,MLPerf Inference v3.1
dense,华为 Ascend 910B,18.0,12.6,14.0,15.8,华为官方基准
moe,NVIDIA A100 80GB,15.0,9.75,80.0,92.0,估算值
moe,NVIDIA H100 80GB,45.0,33.75,80.0,88.0,估算值
```

`CalibrationManager` 初始化时自动加载默认校准数据，无需用户手动导入。

**涉及文件：**
- `cpp/src/estimation_engine.cpp` — 改进 `infer_architecture()`、激活值公式
- `cpp/src/hardware_matcher.cpp` — 改进通信开销模型
- `python/data/calibration_data/default_calibration.csv` — 预置校准数据
- `python/core/calibration_mgr.py` — 自动加载默认校准
- `cpp/tests/test_engine.cpp` — 更新测试用例
- `python/tests/test_integration.py` — 更新集成测试

### 2.3 硬件数据库扩充

在 `python/data/hardware_specs.json` 中补充以下硬件：

| 硬件 | FP16 TFLOPS | INT8 TOPS | 显存 | 带宽 | 来源 |
|------|-------------|-----------|------|------|------|
| 昇腾 910A | 320 | 640 | 32GB HBM2 | 1200 GB/s | 华为官网 |
| 昇腾 910C | 400 | 800 | 64GB HBM2e | 1600 GB/s | 华为官网 |
| 昇腾 310P | 150 | 300 | 24GB LPDDR4x | 100 GB/s | 华为官网 |
| 海光 DCU | 200 | 400 | 64GB HBM2 | 800 GB/s | 海光公开资料 |
| 沐曦 N100 | 200 | 400 | 48GB HBM2e | 600 GB/s | 沐曦公开资料 |

**涉及文件：**
- `python/data/hardware_specs.json` — 添加新硬件条目

### 2.4 文档增强

扩展 `docs/formulas.md`，添加：

1. **详细推导**：每个公式从基本原理推导，附完整计算过程
2. **计算示例**：
   - 示例 1：LLaMA-7B FP16 在 A100 80G 上的显存计算
   - 示例 2：Mixtral 8x7B INT8 在 H100 上的吞吐量估算
   - 示例 3：DLRM-large 在 A100 上的 Embedding 显存计算
3. **参考文献对照表**：每条公式标注来源论文/白皮书
4. **公式验证**：与 MLPerf 基准数据的对比表

**涉及文件：**
- `docs/formulas.md` — 大幅扩展
- `docs/usage.md` — 补充推荐模型使用说明

## 3. 实施顺序

建议按以下顺序实施（可并行的子任务标注 `∥`）：

1. **Task 1-2**: 生成式推荐模型 — C++ 引擎 + 测试
2. **Task 3-4**: 估算精度改进 — 公式优化 + 校准数据预置 ∥ 硬件数据库扩充
3. **Task 5**: 推荐模型 Python 业务层 + Web UI 集成
4. **Task 6**: 文档增强
5. **Task 7**: 全面测试验证

## 4. 预期效果

- **模型覆盖度**：从 4/5 类模型提升到 5/5，加上硬件扩充和文档完善，预期得分从 ~70% 提升到 ~90%
- **度量算法精度**：通过公式改进 + 基准校准，预期误差从 ~30-50% 降低到 ~15-25%
- **代码文档完整性**：补充详细推导和示例，预期得分从 ~80% 提升到 ~95%
