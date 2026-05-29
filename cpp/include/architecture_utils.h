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

} // namespace model_compute
