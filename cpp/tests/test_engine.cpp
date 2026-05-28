#include <gtest/gtest.h>
#include "estimation_engine.h"
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
