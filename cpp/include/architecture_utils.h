#pragma once
#include <cmath>
#include <algorithm>

namespace model_compute {

inline void infer_architecture(double param_b, int& num_layers, int& hidden_dim, int& num_heads) {
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

// MoE-aware architecture inference: infer from per-expert size, not total params
inline void infer_moe_architecture(double total_param_b, int num_experts,
                                    int& num_layers, int& hidden_dim, int& num_heads) {
    // MoE models have the same layer count as a dense model of similar hidden size.
    // Per-expert FFN params ≈ total_params / num_experts
    // Dense model params ≈ per_expert_ffn × 1.6 (attention + embedding overhead)
    double per_expert_b = total_param_b / std::max(1, num_experts);
    double effective_dense_b = per_expert_b * 1.6;
    // Clamp to reasonable range (Mixtral 8x7B: per_expert=5.8B → effective=9.3B → 32 layers)
    effective_dense_b = std::max(1.0, std::min(effective_dense_b, 70.0));
    infer_architecture(effective_dense_b, num_layers, hidden_dim, num_heads);
}

} // namespace model_compute
