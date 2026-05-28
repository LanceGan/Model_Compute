# 模型对算力等效建模评估工具

## 简介
基于异构算力资源池，支持多负载的算力需求表征框架和动态调度工具。

## 支持的模型类型
- 稠密模型 (Dense): LLaMA, GPT, Qwen 等
- MoE 模型: Mixtral, DeepSeek-V2 等
- 类o1推理模型: DeepSeek-R1 等
- 多模态模型: LLaVA, Qwen-VL 等

## 安装
```bash
# 编译 C++ 模块
bash scripts/build.sh

# 安装 Python 依赖
pip install -r requirements.txt
pip install -e python/
```

## 使用
```bash
streamlit run python/web/app.py
```

## 测试
```bash
bash scripts/run_tests.sh
```
