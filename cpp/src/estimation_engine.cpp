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

static void infer_architecture(double param_b, int& num_layers, int& hidden_dim, int& num_heads) {
    // Parameterized formula: P ≈ 12 * L * d² (Transformer params ignoring embedding)
    // With typical ratio d/L ≈ 128 for LLaMA-family models
    double p = param_b * 1e9;
    if (p <= 0) {
        num_layers = 2;
        hidden_dim = 512;
        num_heads = 8;
        return;
    }
    double L = std::pow(p / (12.0 * 128.0 * 128.0), 1.0 / 3.0);
    num_layers = std::max(2, static_cast<int>(std::round(L)));
    hidden_dim = std::max(512, static_cast<int>(std::round(128.0 * num_layers)));
    hidden_dim = (hidden_dim + 63) / 64 * 64;
    num_heads = std::max(1, hidden_dim / 128);
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
        case ModelType::RECOMMENDATION:
            estimate_recommendation(params, result);
            break;
    }
    return result;
}

double EstimationEngine::estimate_dense(const ModelParams& p, EstimationResult& r) {
    double bpp = bytes_per_param(p.quant);
    double params_bytes = p.param_billions * 1e9 * bpp;
    r.weight_memory_gb = params_bytes / 1e9;

    int num_layers, hidden_dim, num_heads;
    infer_architecture(p.param_billions, num_layers, hidden_dim, num_heads);

    int effective_kv_heads = (p.num_kv_heads > 0) ? p.num_kv_heads : num_heads;
    int effective_head_dim = (p.head_dim > 0) ? p.head_dim : (hidden_dim / num_heads);
    int kv_dim = effective_kv_heads * effective_head_dim;
    double kv_bpe = 2.0;  // KV cache always FP16 during inference
    double kv_bytes = 2.0 * num_layers * kv_dim * p.max_tokens * p.concurrency * kv_bpe;
    r.kv_cache_gb = kv_bytes / 1e9;

    // Activation memory: based on architecture dimensions
    // Inference factor ≈ 2 (no backward pass intermediates)
    double activation_factor = 2.0;
    // Inference: activations stored per-layer, not all layers simultaneously
    double activation_bytes = static_cast<double>(p.concurrency) * p.max_tokens * hidden_dim * bpp * activation_factor;

    double total_bytes = params_bytes + kv_bytes + activation_bytes;
    double base_memory_gb = total_bytes / 1e9;

    // Framework overhead
    r.runtime_overhead_gb = 0.8;  // CUDA/PyTorch runtime

    // Fragmentation: larger models have lower fragmentation ratio
    double frag_ratio = (base_memory_gb > 50.0) ? 0.05 : 0.10;
    r.fragmentation_gb = base_memory_gb * frag_ratio;

    r.memory_gb = base_memory_gb + r.fragmentation_gb + r.runtime_overhead_gb;

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

    int num_layers, hidden_dim, num_heads;
    infer_architecture(p.param_billions, num_layers, hidden_dim, num_heads);

    int effective_kv_heads = (p.num_kv_heads > 0) ? p.num_kv_heads : num_heads;
    int effective_head_dim = (p.head_dim > 0) ? p.head_dim : (hidden_dim / num_heads);
    int kv_dim = effective_kv_heads * effective_head_dim;
    double kv_bpe = 2.0;  // KV cache always FP16 during inference
    double kv_bytes = 2.0 * num_layers * kv_dim * p.max_tokens * p.concurrency * kv_bpe;
    r.kv_cache_gb = kv_bytes / 1e9;

    double total_bytes = total_params_bytes + kv_bytes;
    double base_memory_gb = total_bytes / 1e9;

    // Framework overhead
    r.runtime_overhead_gb = 0.8;  // CUDA/PyTorch runtime

    // Fragmentation: larger models have lower fragmentation ratio
    double frag_ratio = (base_memory_gb > 50.0) ? 0.05 : 0.10;
    r.fragmentation_gb = base_memory_gb * frag_ratio;

    r.memory_gb = base_memory_gb + r.fragmentation_gb + r.runtime_overhead_gb;

    int input_seq = p.max_tokens / 2;
    int output_seq = p.max_tokens / 2;
    double base_flops = 2.0 * active_params_b * 1e9 * (input_seq + output_seq);
    double routing_overhead = base_flops * 0.01;
    r.flops_total = (base_flops + routing_overhead) * p.concurrency;

    double bytes_per_token = active_params_b * 1e9 * bpp + 2.0 * num_layers * hidden_dim * bpp;
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

    int num_layers, hidden_dim, num_heads;
    infer_architecture(p.param_billions, num_layers, hidden_dim, num_heads);
    double img_kv_bytes = 2.0 * num_layers * hidden_dim * img_tokens * p.num_images * p.concurrency * bpp;
    double img_kv_gb = img_kv_bytes / 1e9;

    double vit_flops = 2.0 * vit_params_b * 1e9 * img_tokens * p.num_images * p.concurrency;

    r.weight_memory_gb += vit_memory;
    r.kv_cache_gb += img_kv_gb;
    r.memory_gb += vit_memory + img_kv_gb;
    r.flops_total += vit_flops;

    return r.memory_gb;
}

double EstimationEngine::estimate_recommendation(const ModelParams& p, EstimationResult& r) {
    if (p.num_sparse_features <= 0 || p.vocab_size_per_feature <= 0 || p.embed_dim <= 0) {
        return r.memory_gb;  // Invalid config, return zero
    }

    double bpp = bytes_per_param(p.quant);

    if (!p.mlp_dims.empty() && p.num_sparse_features > 0) {
        // DLRM/DeepFM: Embedding-dominant model
        double embedding_bytes = static_cast<double>(p.num_sparse_features)
                                 * p.vocab_size_per_feature * p.embed_dim * bpp;
        r.weight_memory_gb = embedding_bytes / 1e9;

        // MLP parameters: In DLRM, each sparse feature has an independent
        // bottom MLP that transforms its embedding before feature interaction.
        // This is standard DLRM architecture (Naumov et al., arXiv:1906.00091).
        double mlp_params = 0;
        double mlp_flops_per_feature = 0;
        int prev_dim = p.embed_dim;
        for (int dim : p.mlp_dims) {
            mlp_params += static_cast<double>(prev_dim) * dim + dim;
            mlp_flops_per_feature += 2.0 * prev_dim * dim;
            prev_dim = dim;
        }
        double mlp_bytes = mlp_params * p.num_sparse_features * bpp;
        r.weight_memory_gb += mlp_bytes / 1e9;

        r.kv_cache_gb = 0;

        double total_bytes = embedding_bytes + mlp_bytes;
        double base_memory_gb = total_bytes / 1e9;

        // Framework overhead
        r.runtime_overhead_gb = 0.8;  // CUDA/PyTorch runtime

        // Fragmentation: larger models have lower fragmentation ratio
        double frag_ratio = (base_memory_gb > 50.0) ? 0.05 : 0.10;
        r.fragmentation_gb = base_memory_gb * frag_ratio;

        r.memory_gb = base_memory_gb + r.fragmentation_gb + r.runtime_overhead_gb;

        // FLOPs: MLP forward pass per feature
        r.flops_total = mlp_flops_per_feature * p.num_sparse_features * p.concurrency;

        // Bandwidth: per-request estimate (not scaled by concurrency), consistent
        // with how estimate_dense handles bandwidth for a single request.
        // Embedding lookups are random access.
        double lookup_bytes = p.num_sparse_features * p.embed_dim * bpp;
        r.bandwidth_gbs = lookup_bytes * 10.0 / 1e9;

    } else if (p.num_sparse_features > 0) {
        // Sequential recommendation: Dense backbone + item embedding
        if (p.param_billions > 0) {
            ModelParams dense_params = p;
            dense_params.type = ModelType::DENSE;
            estimate_dense(dense_params, r);
        }

        double embedding_bytes = static_cast<double>(p.vocab_size_per_feature)
                                 * p.embed_dim * bpp;
        double embedding_gb = embedding_bytes / 1e9;
        r.weight_memory_gb += embedding_gb;
        r.memory_gb += embedding_gb;
    }

    return r.memory_gb;
}

} // namespace model_compute
