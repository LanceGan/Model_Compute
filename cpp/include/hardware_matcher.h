#pragma once
#include "estimation_engine.h"
#include <string>
#include <vector>

namespace model_compute {

struct HardwareSpec {
    std::string name;
    std::string vendor;
    std::string architecture;
    std::string type;
    double fp16_tflops;
    double int8_tops;
    double fp32_tflops;
    double memory_gb;
    std::string memory_type;
    double memory_bandwidth_gbs;
    double nvlink_bandwidth_gbs;
    std::string pcie_version;
    double max_tdp_watts;
    double cost_per_unit;
};

struct HardwareConfig {
    HardwareSpec hardware;
    int num_cards;
    double estimated_throughput;
    double estimated_latency_ms;
    std::string bottleneck_type;
    std::string parallel_strategy;
    bool meets_baseline;
};

class HardwareMatcher {
public:
    std::vector<HardwareConfig> match(
        const EstimationResult& estimation,
        const ModelParams& model_params,
        const std::vector<HardwareSpec>& hardware_pool,
        double baseline_throughput = 10.0
    );

private:
    int calculate_cards_by_memory(double memory_needed, double card_memory);
    int calculate_cards_by_compute(double flops_needed, double card_tflops);
    double estimate_comm_overhead(int cards, const HardwareSpec& hw, const ModelParams& mp);
    std::string select_parallel_strategy(int cards, ModelType type);
};

} // namespace model_compute
