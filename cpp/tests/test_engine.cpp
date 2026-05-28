#include <gtest/gtest.h>
#include "estimation_engine.h"
#include "hardware_matcher.h"
#include <cmath>

using namespace model_compute;

TEST(DenseEstimation, WeightMemoryFP16) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 14.0, 0.5);
}

TEST(DenseEstimation, WeightMemoryINT8) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::INT8;
    params.concurrency = 1;
    params.max_tokens = 2048;
    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 7.0, 0.5);
}

TEST(DenseEstimation, WeightMemoryINT4) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 70.0;
    params.quant = Quantization::INT4;
    params.concurrency = 1;
    params.max_tokens = 2048;
    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 35.0, 1.0);
}

TEST(DenseEstimation, TotalMemoryIncludesOverhead) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    auto result = engine.estimate(params);
    EXPECT_GT(result.memory_gb, result.weight_memory_gb);
}

TEST(DenseEstimation, KVCacheScalesWithConcurrency) {
    EstimationEngine engine;
    ModelParams p1;
    p1.type = ModelType::DENSE;
    p1.param_billions = 7.0;
    p1.quant = Quantization::FP16;
    p1.concurrency = 1;
    p1.max_tokens = 2048;
    ModelParams p2 = p1;
    p2.concurrency = 16;
    auto r1 = engine.estimate(p1);
    auto r2 = engine.estimate(p2);
    EXPECT_GT(r2.kv_cache_gb, r1.kv_cache_gb);
}

TEST(DenseEstimation, FLOPsCalculation) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    auto result = engine.estimate(params);
    EXPECT_GT(result.flops_total, 0);
}

TEST(DenseEstimation, LargeModel70B) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 70.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 4096;
    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 140.0, 2.0);
    EXPECT_GT(result.memory_gb, 140.0);
}

TEST(HardwareMatcher, SingleCardSufficient) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 10.0;
    est.flops_total = 1e18;
    est.bandwidth_gbs = 50.0;

    ModelParams mp;
    mp.type = ModelType::DENSE;
    mp.param_billions = 7.0;
    mp.concurrency = 1;

    HardwareSpec hw;
    hw.name = "A100 80G";
    hw.memory_gb = 80.0;
    hw.fp16_tflops = 312.0;
    hw.memory_bandwidth_gbs = 2039.0;
    hw.nvlink_bandwidth_gbs = 600.0;
    hw.cost_per_unit = 10000.0;

    auto configs = matcher.match(est, mp, {hw}, 10.0);
    ASSERT_FALSE(configs.empty());
    EXPECT_EQ(configs[0].num_cards, 1);
    EXPECT_TRUE(configs[0].meets_baseline);
}

TEST(HardwareMatcher, NeedsMultipleCardsForMemory) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 150.0;
    est.flops_total = 1e18;
    est.bandwidth_gbs = 50.0;

    ModelParams mp;
    mp.type = ModelType::DENSE;
    mp.param_billions = 70.0;
    mp.concurrency = 1;

    HardwareSpec hw;
    hw.name = "A100 80G";
    hw.memory_gb = 80.0;
    hw.fp16_tflops = 312.0;
    hw.memory_bandwidth_gbs = 2039.0;
    hw.nvlink_bandwidth_gbs = 600.0;
    hw.cost_per_unit = 10000.0;

    auto configs = matcher.match(est, mp, {hw}, 10.0);
    ASSERT_FALSE(configs.empty());
    EXPECT_GE(configs[0].num_cards, 2);
}

TEST(HardwareMatcher, MultipleHardwareOptions) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 30.0;
    est.flops_total = 1e18;
    est.bandwidth_gbs = 50.0;

    ModelParams mp;
    mp.type = ModelType::DENSE;
    mp.param_billions = 13.0;
    mp.concurrency = 1;

    HardwareSpec a100;
    a100.name = "A100 80G";
    a100.memory_gb = 80.0;
    a100.fp16_tflops = 312.0;
    a100.memory_bandwidth_gbs = 2039.0;
    a100.nvlink_bandwidth_gbs = 600.0;
    a100.cost_per_unit = 10000.0;

    HardwareSpec h100;
    h100.name = "H100 80G";
    h100.memory_gb = 80.0;
    h100.fp16_tflops = 990.0;
    h100.memory_bandwidth_gbs = 3350.0;
    h100.nvlink_bandwidth_gbs = 900.0;
    h100.cost_per_unit = 25000.0;

    auto configs = matcher.match(est, mp, {a100, h100}, 10.0);
    EXPECT_GE(configs.size(), 2u);
}
