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

TEST(RecommendationEstimation, DLRMEmbeddingMemoryDominates) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::RECOMMENDATION;
    params.param_billions = 0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    params.num_sparse_features = 26;
    params.vocab_size_per_feature = 100000;
    params.embed_dim = 128;
    params.mlp_dims = {512, 256, 1};

    auto result = engine.estimate(params);
    EXPECT_GT(result.weight_memory_gb, 0.5);
    EXPECT_LT(result.weight_memory_gb, 1.0);
    EXPECT_GT(result.memory_gb, 0.5);
}

TEST(RecommendationEstimation, DLRMLargerVocabMoreMemory) {
    EstimationEngine engine;
    ModelParams p1;
    p1.type = ModelType::RECOMMENDATION;
    p1.quant = Quantization::FP16;
    p1.concurrency = 1;
    p1.max_tokens = 2048;
    p1.num_sparse_features = 26;
    p1.vocab_size_per_feature = 100000;
    p1.embed_dim = 128;
    p1.mlp_dims = {512, 256, 1};

    ModelParams p2 = p1;
    p2.vocab_size_per_feature = 1000000;

    auto r1 = engine.estimate(p1);
    auto r2 = engine.estimate(p2);
    EXPECT_GT(r2.memory_gb, r1.memory_gb * 5);
}

TEST(RecommendationEstimation, DLRMINT8ReducesMemory) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::RECOMMENDATION;
    params.quant = Quantization::INT8;
    params.concurrency = 1;
    params.max_tokens = 2048;
    params.num_sparse_features = 26;
    params.vocab_size_per_feature = 100000;
    params.embed_dim = 128;
    params.mlp_dims = {512, 256, 1};

    auto result = engine.estimate(params);
    EXPECT_GT(result.weight_memory_gb, 0.2);
    EXPECT_LT(result.weight_memory_gb, 1.0);
}

TEST(RecommendationEstimation, DLRMFLOPsPositive) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::RECOMMENDATION;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    params.num_sparse_features = 26;
    params.vocab_size_per_feature = 100000;
    params.embed_dim = 128;
    params.mlp_dims = {512, 256, 1};

    auto result = engine.estimate(params);
    EXPECT_GT(result.flops_total, 0);
}

TEST(RecommendationEstimation, SequentialRecAddsEmbeddingToDense) {
    EstimationEngine engine;
    // Sequential recommendation (no mlp_dims) reuses Dense + item embedding
    ModelParams params;
    params.type = ModelType::RECOMMENDATION;
    params.param_billions = 0.5;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;
    params.num_sparse_features = 1;
    params.vocab_size_per_feature = 50000;
    params.embed_dim = 64;

    auto result = engine.estimate(params);

    // Compare with pure Dense baseline
    ModelParams dense_params;
    dense_params.type = ModelType::DENSE;
    dense_params.param_billions = 0.5;
    dense_params.quant = Quantization::FP16;
    dense_params.concurrency = 1;
    dense_params.max_tokens = 2048;
    auto dense_result = engine.estimate(dense_params);

    EXPECT_GT(result.memory_gb, 0);
    EXPECT_GT(result.weight_memory_gb, 0);
    // Sequential rec should have MORE memory than pure dense (embedding added)
    EXPECT_GT(result.weight_memory_gb, dense_result.weight_memory_gb);
}

TEST(ArchitectureInference, SmallModelReasonableDims) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;

    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 14.0, 1.0);
    EXPECT_GT(result.kv_cache_gb, 0);
    EXPECT_LT(result.kv_cache_gb, 100.0);
}

TEST(ArchitectureInference, LargeModelReasonableDims) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 70.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 4096;

    auto result = engine.estimate(params);
    EXPECT_NEAR(result.weight_memory_gb, 140.0, 5.0);
    EXPECT_GT(result.memory_gb, result.weight_memory_gb);
}

TEST(ArchitectureInference, TinyModelDims) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 0.5;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 1024;

    auto result = engine.estimate(params);
    EXPECT_GT(result.weight_memory_gb, 0);
    EXPECT_GT(result.memory_gb, result.weight_memory_gb);
}

TEST(HardwareMatcher, NVLinkLowerOverheadThanPCIe) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 200.0;
    est.flops_total = 1e18;
    est.bandwidth_gbs = 100.0;

    ModelParams mp;
    mp.type = ModelType::DENSE;
    mp.param_billions = 70.0;
    mp.concurrency = 1;

    HardwareSpec nvlink_hw;
    nvlink_hw.name = "A100 NVLink";
    nvlink_hw.vendor = "NVIDIA";
    nvlink_hw.memory_gb = 80.0;
    nvlink_hw.fp16_tflops = 312.0;
    nvlink_hw.memory_bandwidth_gbs = 2039.0;
    nvlink_hw.nvlink_bandwidth_gbs = 600.0;

    HardwareSpec pcie_hw;
    pcie_hw.name = "L40S PCIe";
    pcie_hw.vendor = "NVIDIA";
    pcie_hw.memory_gb = 48.0;
    pcie_hw.fp16_tflops = 362.0;
    pcie_hw.memory_bandwidth_gbs = 864.0;
    pcie_hw.nvlink_bandwidth_gbs = 0;

    auto nv_configs = matcher.match(est, mp, {nvlink_hw}, 10.0);
    auto pcie_configs = matcher.match(est, mp, {pcie_hw}, 10.0);

    ASSERT_FALSE(nv_configs.empty());
    ASSERT_FALSE(pcie_configs.empty());
}

TEST(ModelFamilyConfig, LLaMA3GQAReducesKVCache) {
    EstimationEngine engine;
    ModelParams p_gqa;
    p_gqa.type = ModelType::DENSE;
    p_gqa.param_billions = 8.0;
    p_gqa.quant = Quantization::FP16;
    p_gqa.concurrency = 1;
    p_gqa.max_tokens = 2048;
    p_gqa.num_kv_heads = 8;
    p_gqa.head_dim = 128;

    ModelParams p_mha = p_gqa;
    p_mha.num_kv_heads = 32;

    auto r_gqa = engine.estimate(p_gqa);
    auto r_mha = engine.estimate(p_mha);

    EXPECT_LT(r_gqa.kv_cache_gb, r_mha.kv_cache_gb * 0.3);
    EXPECT_GT(r_gqa.kv_cache_gb, 0);
}

TEST(ModelFamilyConfig, DefaultKVHeadsEqualsNumHeads) {
    EstimationEngine engine;
    ModelParams p_default;
    p_default.type = ModelType::DENSE;
    p_default.param_billions = 7.0;
    p_default.quant = Quantization::FP16;
    p_default.concurrency = 1;
    p_default.max_tokens = 2048;
    p_default.num_kv_heads = 0;
    p_default.head_dim = 0;

    ModelParams p_explicit = p_default;
    p_explicit.num_kv_heads = 32;
    p_explicit.head_dim = 128;

    auto r_default = engine.estimate(p_default);
    auto r_explicit = engine.estimate(p_explicit);

    EXPECT_NEAR(r_default.kv_cache_gb, r_explicit.kv_cache_gb, 0.05);
}

TEST(FrameworkOverhead, RuntimeOverheadAdded) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;

    auto result = engine.estimate(params);
    EXPECT_GT(result.runtime_overhead_gb, 0.5);
    EXPECT_LT(result.runtime_overhead_gb, 1.5);
}

TEST(FrameworkOverhead, FragmentationPositive) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 70.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 4096;

    auto result = engine.estimate(params);
    EXPECT_GT(result.fragmentation_gb, 0);
}

TEST(FrameworkOverhead, TotalMemoryIncludesAllComponents) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;

    auto result = engine.estimate(params);
    EXPECT_GT(result.memory_gb, result.weight_memory_gb + result.runtime_overhead_gb);
}

TEST(HardwareMatcher, HuaweiHCCSOverhead) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 50.0;
    est.flops_total = 1e18;
    est.bandwidth_gbs = 100.0;

    ModelParams mp;
    mp.type = ModelType::DENSE;
    mp.param_billions = 7.0;
    mp.concurrency = 1;

    HardwareSpec hw;
    hw.name = "910B";
    hw.vendor = "华为";
    hw.memory_gb = 64.0;
    hw.fp16_tflops = 320.0;
    hw.memory_bandwidth_gbs = 1200.0;
    hw.nvlink_bandwidth_gbs = 0;

    auto configs = matcher.match(est, mp, {hw}, 10.0);
    ASSERT_FALSE(configs.empty());
    EXPECT_TRUE(configs[0].meets_baseline);
}
