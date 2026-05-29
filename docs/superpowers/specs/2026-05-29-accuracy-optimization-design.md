# 估算精度优化设计文档

## 1. 优化目标

针对赛题"度量算法评估（40%）"评分项，通过三个维度提升估算精度：
- **模型族特化**：针对不同模型族的架构特点使用精确公式
- **框架开销建模**：考虑运行时固定开销和内存碎片化
- **校准数据扩展 + 验证**：扩充校准数据，建立验证框架量化误差

## 2. 模型族特化

### 2.1 架构元数据扩展

在 `ModelParams` 中新增架构标识字段：

```cpp
int num_kv_heads = 0;    // KV head 数（GQA: < num_heads, MHA: = num_heads）
int head_dim = 0;        // 每个 head 的维度
bool use_swiglu = false; // 是否使用 SwiGLU FFN
```

### 2.2 模型族配置表

在 C++ 引擎中内置已知模型族的精确配置：

| 模型 | layers | hidden | heads | kv_heads | head_dim | SwiGLU |
|------|--------|--------|-------|----------|----------|--------|
| LLaMA-2 7B | 32 | 4096 | 32 | 32 | 128 | No |
| LLaMA-2 13B | 40 | 5120 | 40 | 40 | 128 | No |
| LLaMA-2 70B | 80 | 8192 | 64 | 64 | 128 | No |
| LLaMA-3 8B | 32 | 4096 | 32 | 8 | 128 | Yes |
| LLaMA-3 70B | 80 | 8192 | 64 | 8 | 128 | Yes |
| Qwen-2.5 7B | 28 | 3584 | 28 | 4 | 128 | Yes |
| Qwen-2.5 72B | 80 | 8192 | 64 | 8 | 128 | Yes |
| DeepSeek-V3 | 61 | 7168 | 128 | 1 | 128 | Yes |

### 2.3 公式修正

**KV Cache（GQA/MQA 支持）：**
```
原：KV Cache = 2 × num_layers × hidden_dim × seq_len × batch × bpp
新：KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × bpp
```

**参数量（SwiGLU）：**
```
标准 FFN: 2 × hidden_dim × ffn_dim
SwiGLU:   3 × hidden_dim × ffn_dim  （+33%）
```

**涉及文件：**
- `cpp/include/estimation_engine.h` — 新增字段
- `cpp/src/estimation_engine.cpp` — 修正公式，新增配置表
- `python/data/model_presets.json` — 添加新模型预设
- `python/core/model_analyzer.py` — 支持新字段
- `cpp/bindings/pybind_module.cpp` — 暴露新字段
- `cpp/tests/test_engine.cpp` — GQA/SwiGLU 测试
- `python/tests/test_integration.py` — 新预设集成测试

## 3. 框架开销建模

### 3.1 运行时固定开销

```
CUDA/PyTorch 运行时 ≈ 0.8 GB（固定）
```

### 3.2 内存碎片化

```
碎片化 = 模型显存 × fragmentation_ratio
  大模型（>50GB）: 0.05
  小模型（<10GB）: 0.10
```

### 3.3 总显存公式修正

```
原：总显存 = (权重 + KV Cache + 激活值) × 1.10
新：总显存 = (权重 + KV Cache + 激活值) × (1 + fragmentation) + runtime_overhead
```

### 3.4 新增字段

```cpp
struct EstimationResult {
    // ... existing fields ...
    double runtime_overhead_gb;    // 运行时固定开销
    double fragmentation_gb;       // 内存碎片化开销
};
```

**涉及文件：**
- `cpp/include/estimation_engine.h` — 新增字段
- `cpp/src/estimation_engine.cpp` — 修正公式
- `cpp/bindings/pybind_module.cpp` — 暴露新字段
- `cpp/tests/test_engine.cpp` — 测试

## 4. 校准数据扩展 + 验证框架

### 4.1 校准数据扩展

扩充 `python/data/calibration_data/default_calibration.csv` 至 25+ 条：

- MLPerf Inference v3.1/v4.0 数据（LLaMA-2 on A100/H100）
- 华为官方基准（7B/13B/70B on 910B）
- 技术博客/论文数据（Mixtral, DeepSeek）
- 估算值（标注来源）

### 4.2 验证框架

创建 `python/tests/test_validation.py`，对比预测值与已知基准：

```python
class TestValidation:
    def test_llama2_7b_a100_accuracy(self):
        """预测值应在实际值的 ±30% 以内"""

    def test_llama2_70b_a100_multi_card(self):
        """70B on 4x A100 验证"""

    def test_mixtral_8x7b_memory(self):
        """Mixtral 显存验证"""
```

### 4.3 MAPE 计算脚本

创建 `scripts/validate_accuracy.py`：
- 读取校准数据
- 计算预测值 vs 实际值的 MAPE
- 输出按模型类型/硬件的分项 MAPE
- 生成验证报告

**涉及文件：**
- `python/data/calibration_data/default_calibration.csv` — 扩充数据
- `python/tests/test_validation.py` — 验证测试
- `scripts/validate_accuracy.py` — MAPE 计算脚本

## 5. 实施顺序

1. **Task 1-3**: 模型族特化 — C++ 引擎 + pybind11 + 测试
2. **Task 4**: 框架开销建模
3. **Task 5**: 新模型预设（LLaMA-3、Qwen-2.5 等）
4. **Task 6**: 校准数据扩展
5. **Task 7**: 验证框架 + MAPE 脚本
6. **Task 8**: 文档更新
7. **Task 9**: 最终验证

## 6. 预期效果

- **度量算法精度**：MAPE 从 ~30-50% 降低到 ~15-25%
- **模型覆盖度**：新增 LLaMA-3、Qwen-2.5、DeepSeek-V3 等热门模型预设
- **可验证性**：验证框架提供量化的精度指标
