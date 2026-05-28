# 估算公式

## Dense 模型

### 显存
- 模型权重 = 参数量 × 每参数字节数 (FP16=2, INT8=1, INT4=0.5)
- KV Cache = 2 × 层数 × hidden_dim × seq_len × 并发量 × 每参数字节数
- 总显存 = (权重 + KV Cache + 激活值) × 1.10

### FLOPs
- Prefill = 2 × 参数量 × 输入序列长度
- Decode (每token) = 2 × 参数量

### 吞吐量
- 计算受限 = TFLOPS × 10^12 / (2 × 参数量)
- 带宽受限 = 带宽 × 10^9 / (参数量 × 每参数字节数)

## MoE 模型
- 显存按总参数量计算
- FLOPs 按激活参数量计算

## 类o1推理模型
- 在 Dense 基础上乘以推理 token 倍率 (2x~10x)

## 多模态模型
- 语言模型 + 视觉编码器分开估算

## 参考文献
- Megatron-LM (arXiv:1909.08053)
- Switch Transformers (Fedus et al., 2022)
- DeepSeek-V2 (arXiv:2405.04434)
