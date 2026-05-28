# 使用说明

## 安装

```bash
# 1. 编译 C++ 模块
bash scripts/build.sh

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装项目
pip install -e python/
```

## 启动 Web UI

```bash
streamlit run python/web/app.py
```

浏览器打开 http://localhost:8501

## 功能说明

### 算力估算
输入模型参数，获取推荐硬件配置和性能预估。

### 生成式推荐模型

支持 DLRM/DeepFM 和序列推荐 Transformer：

```python
# DLRM 模型估算
params = analyzer.create_params(
    model_type="recommendation",
    preset_name="DLRM-small",
    quant="FP16",
    concurrency=1,
    max_tokens=2048,
)

# 自定义推荐模型
params = analyzer.create_params(
    model_type="recommendation",
    num_sparse_features=26,
    vocab_size_per_feature=100000,
    embed_dim=128,
    mlp_dims=[512, 256, 1],
    quant="INT8",
    concurrency=1,
    max_tokens=2048,
)
```

支持的预设模型：
- DLRM-small: Facebook DLRM 基准配置 (26特征, 100K词表)
- DLRM-large: 大规模推荐模型 (26特征, 1M词表)
- DeepFM: DeepFM 基准配置 (39特征, 500K词表)
- SASRec: 序列推荐 Transformer (50K物品词表)

### 多硬件对比
同时对比所有可用硬件的性能表现。

### 敏感性分析
分析并发量/参数量/序列长度变化对性能的影响。

### 管理
管理硬件数据库和校准数据。

## Python API

```python
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB

import model_compute as mc

analyzer = ModelAnalyzer()
params = analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)

engine = mc.EstimationEngine()
result = engine.estimate(params)
print(f"显存需求: {result.memory_gb:.1f} GB")
```
