# 系统架构

## 三层架构

本工具采用 C++ + Python 混合架构：

1. **计算引擎层 (C++)**: FLOPs/显存/带宽计算、硬件匹配、校准拟合
2. **业务逻辑层 (Python)**: 模型分析、数据管理、参数验证
3. **展示层 (Streamlit)**: Web UI、图表、报告导出

通过 pybind11 桥接 C++ 和 Python。

## 目录结构

- `cpp/` — C++ 核心引擎源码
- `python/` — Python 业务层和 Web UI
- `docs/` — 项目文档
- `scripts/` — 构建和测试脚本
