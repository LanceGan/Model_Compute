# 估算公式

## 1. Dense 模型

### 1.1 显存估算

**模型权重：**
```
权重显存 = 参数量 × 每参数字节数
  FP16: 2 bytes/param
  INT8: 1 byte/param
  INT4: 0.5 bytes/param
```

**KV Cache：**
```
KV Cache = 2 × num_layers × hidden_dim × seq_len × concurrency × 每参数字节数
  其中 2 代表 Key 和 Value 两个缓存
```

**激活值：**
```
激活值 = concurrency × max_tokens × hidden_dim × num_layers × 每参数字节数 × activation_factor
  推理模式 activation_factor ≈ 2
```

**总显存：**
```
总显存 = (权重 + KV Cache + 激活值) × 1.10
  1.10 为运行时开销系数（~10%）
```

**架构推断：**
```
基于参数量推导 hidden_dim 和 num_layers：
  P ≈ 12 × L × d²  （Transformer 参数量公式，忽略 embedding）
  d/L ≈ 128         （LLaMA 系列典型比例）
  → L = (P / (12 × 128²))^(1/3)
  → d = 128 × L，对齐到 64 的倍数
```

### 1.2 FLOPs 估算

```
Prefill 阶段 = 2 × 参数量 × 输入序列长度
  （矩阵乘法：每个参数参与 2 次浮点运算）

Decode 阶段（每 token）= 2 × 参数量

总 FLOPs = (Prefill FLOPs + Decode FLOPs × 输出 token 数) × 并发量
```

### 1.3 吞吐量瓶颈分析

```
计算受限吞吐:
  tokens/s = 算力(TFLOPS) × 10^12 / (2 × 参数量)

带宽受限吞吐:
  tokens/s = 带宽(GB/s) × 10^9 / (参数量 × 每参数字节数 + KV_per_token)

实际吞吐 = min(计算受限, 带宽受限)
```

### 1.4 GQA/MQA 注意力优化

现代模型（LLaMA-3、Qwen-2.5）使用 Grouped Query Attention (GQA)：
```
标准 MHA: KV heads = Q heads（如 32 个 KV head）
GQA:      KV heads < Q heads（如 8 个 KV head）

KV Cache 减少比例 = num_kv_heads / num_heads
  LLaMA-3 8B: 8/32 = 0.25x（减少 75%）
  Qwen-2.5 7B: 4/28 = 0.14x（减少 86%）
```

### 1.5 SwiGLU FFN

SwiGLU 激活函数使用三个线性变换（gate + up + down）：
```
标准 FFN: 2 × hidden_dim × ffn_dim 参数
SwiGLU:   3 × hidden_dim × ffn_dim 参数（+33%）
```

### 1.6 框架开销

```
运行时固定开销 ≈ 0.8 GB（CUDA/PyTorch runtime）
内存碎片化 = 基础显存 × fragmentation_ratio
  大模型（>50GB）: 5%
  小模型（<10GB）: 10%
总显存 = 基础显存 + 碎片化 + 运行时开销
```

### 1.7 计算示例：LLaMA-7B FP16 在 A100 80G

```
输入：参数量=7B，量化=FP16，seq_len=2048，并发=1

权重显存 = 7×10⁹ × 2 = 14 GB
架构推断：L=32, d=4096
KV Cache = 2 × 32 × 4096 × 2048 × 1 × 2 = 1.07 GB
激活值 = 1 × 2048 × 4096 × 32 × 2 × 2 = 1.07 GB
总显存 = (14 + 1.07 + 1.07) × 1.10 = 17.8 GB

FLOPs = (2×7×10⁹×1024 + 2×7×10⁹×1024) × 1 = 2.87×10¹³

计算受限吞吐 = 312×10¹² / (2×7×10⁹) = 22.3 tokens/s
带宽受限吞吐 = 2039×10⁹ / (7×10⁹×2 + 2×32×4096×2) = 68.5 tokens/s
实际吞吐 ≈ 22.3 tokens/s（计算受限）
```

## 2. MoE 模型

```
总参数量 = 所有专家权重之和 + 共享层参数
激活参数量 = 每token激活专家数 × 单专家参数 + 共享层参数

显存 = 总参数量 × 每参数字节数 + KV Cache + 开销
FLOPs = 激活参数量 × 2 × tokens × 并发量 + 路由开销(1%)
```

## 3. 类o1推理模型

```
实际输出 token 数 = 用户可见输出 + 内部推理 token
推理 token 倍率：轻度 2-3x，中度 3-5x，重度 5-10x
总计算量 = Dense 基础公式 × (1 + 推理token倍率)
```

## 4. 多模态模型

```
显存 = 语言模型显存 + 视觉编码器显存 + 图像token的KV Cache
图像token数 = (图像分辨率 / patch_size)²
FLOPs = 语言模型FLOPs + 视觉编码器FLOPs × 图像数量
```

## 5. 生成式推荐模型

### 5.1 DLRM/DeepFM

```
Embedding 显存 = num_sparse_features × vocab_size × embed_dim × 每参数字节数
MLP 参数量 = Σ(mlp_dims[i] × mlp_dims[i+1] + mlp_dims[i+1]) × num_sparse_features
总显存 = (Embedding + MLP) × 1.10

FLOPs = 2 × Σ(mlp_dims[i] × mlp_dims[i+1]) × num_sparse_features × concurrency
```

**计算示例：DLRM-small FP16**
```
输入：26特征，词表100K，embed_dim=128，MLP=[512,256,1]

Embedding = 26 × 100000 × 128 × 2 = 665.6 MB = 0.635 GB
MLP = 26 × (512×256+256 + 256×1+1) × 2 = 6.96 MB
总显存 = (0.635 + 0.007) × 1.10 = 0.707 GB

FLOPs = 2 × (512×256 + 256×1) × 26 × 1 = 26.7×10⁶
```

### 5.2 序列推荐 Transformer

```
复用 Dense 估算 + Item Embedding 额外显存
Item Embedding = vocab_size × embed_dim × 每参数字节数
```

## 6. 参考文献

| 公式来源 | 论文/资料 |
|----------|----------|
| Transformer 参数量公式 | Megatron-LM (arXiv:1909.08053) |
| MoE 稀疏激活 | Switch Transformers (Fedus et al., JMLR 2022) |
| MoE 架构 | DeepSeek-V2 (arXiv:2405.04434) |
| 多模态架构 | LLaVA (Liu et al., NeurIPS 2023) |
| 多模态架构 | Qwen-VL (Bai et al., arXiv:2308.12966) |
| DLRM 架构 | DLRM (Naumov et al., arXiv:1906.00091) |
| GPU 架构 | NVIDIA A100 Tensor Core GPU Architecture Whitepaper |
| NPU 架构 | 华为 Ascend 910B 技术规格 |
| 并行策略 | Alpa (Zheng et al., OSDI 2022) |
| 推理优化 | vLLM (Kwon et al., SOSP 2023) |
