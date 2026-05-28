#include "estimation_engine.h"
#include <cmath>
#include <algorithm>

namespace model_compute {

double EstimationEngine::bytes_per_param(Quantization q) {
    switch (q) {
        case Quantization::FP16: return 2.0;
        case Quantization::INT8:  return 1.0;
        case Quantization::INT4:  return 0.5;
        default: return 2.0;
    }
}

static void infer_architecture(double param_b, int& num_layers, int& hidden_dim) {
    if (param_b <= 1.5)      { num_layers = 22;  hidden_dim = 2048; }
    else if (param_b <= 3.5) { num_layers = 26;  hidden_dim = 3200; }
    else if (param_b <= 8.0) { num_layers = 32;  hidden_dim = 4096; }
    else if (param_b <= 15.0){ num_layers = 40;  hidden_dim = 5120; }
    else if (param_b <= 35.0){ num_layers = 60;  hidden_dim = 6656; }
    else if (param_b <= 75.0){ num_layers = 80;  hidden_dim = 8192; }
    else                     { num_layers = 96;  hidden_dim = 12288; }
}

EstimationResult EstimationEngine::estimate(const ModelParams& params) {
    EstimationResult result = {};
    switch (params.type) {
        case ModelType::DENSE:
            estimate_dense(params, result);
            break;
        case ModelType::MOE:
            estimate_moe(params, result);
            break;
        case ModelType::O1_REASONING:
            estimate_o1(params, result);
            break;
        case ModelType::MULTIMODAL:
            estimate_multimodal(params, result);
            break;
    }
    return result;
}

double EstimationEngine::estimate_dense(const ModelParams& p, EstimationResult& r) {
    double bpp = bytes_per_param(p.quant);
    double params_bytes = p.param_billions * 1e9 * bpp;
    r.weight_memory_gb = params_bytes / 1e9;

    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);

    double kv_bytes = 2.0 * num_layers * hidden_dim * p.max_tokens * p.concurrency * bpp;
    r.kv_cache_gb = kv_bytes / 1e9;

    double activation_ratio = 0.02 * std::min(static_cast<double>(p.concurrency), 32.0);
    double activation_bytes = p.param_billions * 1e9 * activation_ratio * bpp;

    double total_bytes = params_bytes + kv_bytes + activation_bytes;
    r.memory_gb = total_bytes * 1.10 / 1e9;

    int input_seq = p.max_tokens / 2;
    int output_seq = p.max_tokens / 2;
    double prefill_flops = 2.0 * p.param_billions * 1e9 * input_seq;
    double decode_flops = 2.0 * p.param_billions * 1e9 * output_seq;
    r.flops_total = (prefill_flops + decode_flops) * p.concurrency;

    double bytes_per_token = p.param_billions * 1e9 * bpp + 2.0 * num_layers * hidden_dim * bpp;
    r.bandwidth_gbs = bytes_per_token * 10.0 / 1e9;

    return r.memory_gb;
}

double EstimationEngine::estimate_moe(const ModelParams& p, EstimationResult& r) {
    double bpp = bytes_per_param(p.quant);
    int num_experts = p.num_experts > 0 ? p.num_experts : 8;
    int active_experts = p.active_experts > 0 ? p.active_experts : 2;

    double total_params_bytes = p.param_billions * 1e9 * bpp;
    r.weight_memory_gb = total_params_bytes / 1e9;

    double active_ratio = static_cast<double>(active_experts) / num_experts;
    double active_params_b = p.param_billions * active_ratio;

    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);

    double kv_bytes = 2.0 * num_layers * hidden_dim * p.max_tokens * p.concurrency * bpp;
    r.kv_cache_gb = kv_bytes / 1e9;

    double total_bytes = total_params_bytes + kv_bytes;
    r.memory_gb = total_bytes * 1.10 / 1e9;

    int input_seq = p.max_tokens / 2;
    int output_seq = p.max_tokens / 2;
    double base_flops = 2.0 * active_params_b * 1e9 * (input_seq + output_seq);
    double routing_overhead = base_flops * 0.01;
    r.flops_total = (base_flops + routing_overhead) * p.concurrency;

    double bytes_per_token = p.param_billions * 1e9 * bpp + 2.0 * num_layers * hidden_dim * bpp;
    r.bandwidth_gbs = bytes_per_token * 10.0 / 1e9;

    return r.memory_gb;
}

double EstimationEngine::estimate_o1(const ModelParams& p, EstimationResult& r) {
    ModelParams dense_params = p;
    dense_params.type = ModelType::DENSE;

    double reasoning_multiplier = 1.0;
    switch (p.reasoning_depth) {
        case 1: reasoning_multiplier = 2.5; break;
        case 2: reasoning_multiplier = 4.0; break;
        case 3: reasoning_multiplier = 7.5; break;
        default: reasoning_multiplier = 1.0; break;
    }

    dense_params.max_tokens = static_cast<int>(p.max_tokens * (1.0 + reasoning_multiplier));
    estimate_dense(dense_params, r);
    return r.memory_gb;
}

double EstimationEngine::estimate_multimodal(const ModelParams& p, EstimationResult& r) {
    ModelParams dense_params = p;
    dense_params.type = ModelType::DENSE;
    estimate_dense(dense_params, r);

    double vit_params_b = 0.304;
    if (p.param_billions > 30.0) vit_params_b = 6.0;

    double bpp = bytes_per_param(p.quant);
    double vit_memory = vit_params_b * 1e9 * bpp / 1e9;

    int img_tokens = 0;
    if (p.image_resolution > 0) {
        int patches = p.image_resolution / 14;
        img_tokens = patches * patches;
    }

    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);
    double img_kv_bytes = 2.0 * num_layers * hidden_dim * img_tokens * p.num_images * p.concurrency * bpp;
    double img_kv_gb = img_kv_bytes / 1e9;

    double vit_flops = 2.0 * vit_params_b * 1e9 * img_tokens * p.num_images * p.concurrency;

    r.weight_memory_gb += vit_memory;
    r.kv_cache_gb += img_kv_gb;
    r.memory_gb += vit_memory + img_kv_gb;
    r.flops_total += vit_flops;

    return r.memory_gb;
}

} // namespace model_compute
