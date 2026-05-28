#pragma once
#include <string>
#include <vector>

namespace model_compute {

enum class ModelType { DENSE, MOE, O1_REASONING, MULTIMODAL };
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
};

struct EstimationResult {
    double memory_gb;
    double flops_total;
    double bandwidth_gbs;
    double kv_cache_gb;
    double weight_memory_gb;
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
};

} // namespace model_compute
