#include "hardware_matcher.h"
#include <algorithm>
#include <cmath>

namespace model_compute {

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
        int cards_compute = calculate_cards_by_compute(estimation.flops_total, hw.fp16_tflops * 1e12);
        config.num_cards = std::max(cards_mem, cards_compute);

        config.estimated_throughput = hw.fp16_tflops * 1e12 * config.num_cards / estimation.flops_total;
        config.estimated_latency_ms = 1000.0 / config.estimated_throughput;

        config.bottleneck_type = (cards_mem > cards_compute) ? "memory" : "compute";
        config.parallel_strategy = select_parallel_strategy(config.num_cards, model_params.type);
        config.meets_baseline = config.estimated_throughput >= baseline_throughput;

        results.push_back(config);
    }
    return results;
}

int HardwareMatcher::calculate_cards_by_memory(double memory_needed, double card_memory) {
    return static_cast<int>(std::ceil(memory_needed / card_memory));
}

int HardwareMatcher::calculate_cards_by_compute(double flops_needed, double card_tflops) {
    return static_cast<int>(std::ceil(flops_needed / card_tflops));
}

double HardwareMatcher::estimate_comm_overhead(int cards, const HardwareSpec& hw, const ModelParams& mp) {
    if (cards <= 1) return 0.0;
    return 0.1 * (cards - 1);
}

std::string HardwareMatcher::select_parallel_strategy(int cards, ModelType type) {
    if (cards <= 1) return "none";
    if (type == ModelType::MOE) return "expert_parallel";
    if (cards <= 8) return "tensor_parallel";
    return "pipeline_parallel";
}

} // namespace model_compute
