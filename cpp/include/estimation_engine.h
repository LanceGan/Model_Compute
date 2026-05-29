#pragma once
#include <string>
#include <vector>

namespace model_compute {

enum class ModelType { DENSE, MOE, O1_REASONING, MULTIMODAL, RECOMMENDATION };
enum class Quantization { FP16, INT8, INT4 };

struct ModelParams {
    ModelType type;
    double param_billions;
    Quantization quant;
    int concurrency;
    int max_tokens;
    int num_experts = 0;
    int active_experts = 0;
    int reasoning_depth = 0;
    int image_resolution = 0;
    int num_images = 1;

    // Recommendation model fields
    int num_sparse_features = 0;
    int vocab_size_per_feature = 0;
    int embed_dim = 0;
    std::vector<int> mlp_dims;

    // Architecture metadata for accurate estimation
    int num_kv_heads = 0;    // GQA: < num_heads, MHA: = num_heads (0 = auto)
    int head_dim = 0;        // Per-head dimension (0 = auto)
    bool use_swiglu = false; // SwiGLU FFN
};

struct EstimationResult {
    double memory_gb;
    double flops_total;
    double bandwidth_gbs;
    double kv_cache_gb;
    double weight_memory_gb;
    double runtime_overhead_gb;
    double fragmentation_gb;
};

class EstimationEngine {
public:
    EstimationResult estimate(const ModelParams& params);

private:
    double bytes_per_param(Quantization q);
    double estimate_dense(const ModelParams& p, EstimationResult& r);
    double estimate_moe(const ModelParams& p, EstimationResult& r);
    double estimate_o1(const ModelParams& p, EstimationResult& r);
    double estimate_multimodal(const ModelParams& p, EstimationResult& r);
    double estimate_recommendation(const ModelParams& p, EstimationResult& r);
};

} // namespace model_compute
