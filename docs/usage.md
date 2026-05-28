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
