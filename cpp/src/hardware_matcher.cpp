#include "hardware_matcher.h"
#include <cmath>
#include <algorithm>

namespace model_compute {

int HardwareMatcher::calculate_cards_by_memory(double memory_needed, double card_memory) {
    return static_cast<int>(std::ceil(memory_needed / card_memory));
}

int HardwareMatcher::calculate_cards_by_compute(double flops_needed, double card_tflops) {
    double card_flops = card_tflops * 1e12;
    return static_cast<int>(std::ceil(flops_needed / card_flops));
}

std::string HardwareMatcher::select_parallel_strategy(int cards, ModelType type) {
    if (cards <= 1) return "none";
    if (cards <= 8) return "TP";
    return "TP+PP";
}

double HardwareMatcher::estimate_comm_overhead(int cards, const HardwareSpec& hw, const ModelParams& mp) {
    if (cards <= 1) return 0.0;
    double tp_overhead = 0.0;
    if (hw.nvlink_bandwidth_gbs > 0) {
        tp_overhead = 0.05 * (cards - 1) / cards;
    } else {
        tp_overhead = 0.10 * (cards - 1) / cards;
    }
    return std::min(tp_overhead, 0.30);
}

std::vector<HardwareConfig> HardwareMatcher::match(
    const EstimationResult& estimation,
    const ModelParams& model_params,
    const std::vector<HardwareSpec>& hardware_pool,
    double baseline_throughput
) {
    std::vector<HardwareConfig> results;

    for (const auto& hw : hardware_pool) {
        HardwareConfig config;
        config.hardware = hw;

        int cards_mem = calculate_cards_by_memory(estimation.memory_gb, hw.memory_gb);
        int cards_compute = calculate_cards_by_compute(estimation.flops_total, hw.fp16_tflops);
        config.num_cards = std::max({cards_mem, cards_compute, 1});

        config.parallel_strategy = select_parallel_strategy(config.num_cards, model_params.type);

        double comm_overhead = estimate_comm_overhead(config.num_cards, hw, model_params);

        double effective_bandwidth = hw.memory_bandwidth_gbs * config.num_cards * (1.0 - comm_overhead);
        double bytes_per_token = 0;
        if (model_params.param_billions > 0) {
            double bpp = (model_params.quant == Quantization::INT8) ? 1.0 :
                         (model_params.quant == Quantization::INT4) ? 0.5 : 2.0;
            bytes_per_token = model_params.param_billions * 1e9 * bpp / config.num_cards;
        }
        if (bytes_per_token > 0) {
            config.estimated_throughput = effective_bandwidth / (bytes_per_token / 1e9);
        } else {
            config.estimated_throughput = 100.0;
        }

        double compute_tflops = (model_params.quant == Quantization::INT8 || model_params.quant == Quantization::INT4)
                                ? hw.int8_tops : hw.fp16_tflops;
        double compute_throughput = (model_params.param_billions > 0)
            ? compute_tflops * 1e12 / (2.0 * model_params.param_billions * 1e9)
            : 1e18;
        config.bottleneck_type = (config.estimated_throughput < compute_throughput) ? "memory" : "compute";
        config.estimated_throughput = std::min(config.estimated_throughput, compute_throughput);

        config.estimated_latency_ms = 1000.0 / config.estimated_throughput;

        config.meets_baseline = config.estimated_throughput >= baseline_throughput;

        results.push_back(config);
    }

    std::sort(results.begin(), results.end(),
        [](const HardwareConfig& a, const HardwareConfig& b) {
            return a.estimated_throughput > b.estimated_throughput;
        });

    return results;
}

} // namespace model_compute
