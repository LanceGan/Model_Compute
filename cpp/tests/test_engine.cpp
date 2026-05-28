#include <gtest/gtest.h>
#include "estimation_engine.h"
#include "hardware_matcher.h"
#include "calibration.h"
#include <cmath>
#include <cstdlib>
#include <string>

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
    est.flops_total = 1e14;
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

TEST(Calibration, AddPointAndRetrieve) {
    Calibration cal;
    CalibrationPoint pt;
    pt.model_type = "dense";
    pt.hardware_name = "A100 80G";
    pt.predicted_throughput = 20.0;
    pt.actual_throughput = 16.0;
    pt.predicted_memory = 14.0;
    pt.actual_memory = 15.5;

    cal.add_point(pt);
    auto factor = cal.get_factor("dense", "A100 80G");

    EXPECT_NEAR(factor.throughput_factor, 0.8, 0.01);  // 16/20
    EXPECT_NEAR(factor.memory_factor, 1.107, 0.01);    // 15.5/14
    EXPECT_EQ(factor.num_points, 1);
}

TEST(Calibration, MultiplePointsAverage) {
    Calibration cal;

    CalibrationPoint pt1;
    pt1.model_type = "dense";
    pt1.hardware_name = "A100 80G";
    pt1.predicted_throughput = 20.0;
    pt1.actual_throughput = 16.0;
    pt1.predicted_memory = 14.0;
    pt1.actual_memory = 15.0;

    CalibrationPoint pt2;
    pt2.model_type = "dense";
    pt2.hardware_name = "A100 80G";
    pt2.predicted_throughput = 30.0;
    pt2.actual_throughput = 25.5;
    pt2.predicted_memory = 28.0;
    pt2.actual_memory = 30.0;

    cal.add_point(pt1);
    cal.add_point(pt2);
    auto factor = cal.get_factor("dense", "A100 80G");

    // Average of (16/20=0.8) and (25.5/30=0.85) = 0.825
    EXPECT_NEAR(factor.throughput_factor, 0.825, 0.01);
    EXPECT_EQ(factor.num_points, 2);
}

TEST(Calibration, UnknownKeyReturnsDefault) {
    Calibration cal;
    auto factor = cal.get_factor("moe", "H100");
    EXPECT_NEAR(factor.throughput_factor, 1.0, 0.001);
    EXPECT_NEAR(factor.memory_factor, 1.0, 0.001);
    EXPECT_EQ(factor.num_points, 0);
}

TEST(Calibration, AdjustThroughput) {
    Calibration cal;
    CalibrationPoint pt;
    pt.model_type = "dense";
    pt.hardware_name = "A100 80G";
    pt.predicted_throughput = 20.0;
    pt.actual_throughput = 16.0;
    pt.predicted_memory = 14.0;
    pt.actual_memory = 15.0;
    cal.add_point(pt);

    double adjusted = cal.adjust_throughput(25.0, "dense", "A100 80G");
    EXPECT_NEAR(adjusted, 20.0, 0.1);  // 25 * 0.8
}

TEST(Calibration, SaveAndLoad) {
    Calibration cal;
    CalibrationPoint pt;
    pt.model_type = "dense";
    pt.hardware_name = "A100 80G";
    pt.predicted_throughput = 20.0;
    pt.actual_throughput = 16.0;
    pt.predicted_memory = 14.0;
    pt.actual_memory = 15.0;
    cal.add_point(pt);

    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
    std::string path = std::string(tmp) + "/test_calibration.csv";
    cal.save_to_file(path);

    Calibration cal2;
    cal2.load_from_file(path);
    auto factor = cal2.get_factor("dense", "A100 80G");
    EXPECT_NEAR(factor.throughput_factor, 0.8, 0.01);

    std::remove(path.c_str());
}
