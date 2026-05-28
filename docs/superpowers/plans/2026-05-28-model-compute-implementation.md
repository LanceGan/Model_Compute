# 模型对算力等效建模评估工具 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 C++ + Python 混合架构的算力等效建模评估工具，支持 Dense/MoE/o1/多模态四类模型，通过 pybind11 桥接，提供 Streamlit Web UI。

**Architecture:** C++ 核心引擎负责 FLOPs/显存/带宽计算和硬件匹配搜索；Python 业务层负责模型分析、数据管理和校准；Streamlit 提供 Web 交互界面。通过 pybind11 将 C++ 引擎暴露给 Python。

**Tech Stack:** C++17, pybind11, CMake, Python 3.10+, Streamlit, Google Test, pytest, JSON

---

## Task 1: 项目脚手架搭建

**Files:**
- Create: `CMakeLists.txt`
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/include/estimation_engine.h`
- Create: `cpp/include/hardware_matcher.h`
- Create: `cpp/include/calibration.h`
- Create: `cpp/src/estimation_engine.cpp`
- Create: `cpp/src/hardware_matcher.cpp`
- Create: `cpp/src/calibration.cpp`
- Create: `cpp/bindings/pybind_module.cpp`
- Create: `cpp/tests/test_engine.cpp`
- Create: `python/__init__.py`
- Create: `python/core/__init__.py`
- Create: `python/core/model_analyzer.py`
- Create: `python/core/hardware_db.py`
- Create: `python/core/calibration_mgr.py`
- Create: `python/data/hardware_specs.json`
- Create: `python/data/model_presets.json`
- Create: `python/data/calibration_data/.gitkeep`
- Create: `python/web/__init__.py`
- Create: `python/web/app.py`
- Create: `python/web/pages/__init__.py`
- Create: `python/web/pages/estimation.py`
- Create: `python/web/pages/comparison.py`
- Create: `python/web/pages/sensitivity.py`
- Create: `python/web/pages/management.py`
- Create: `python/web/components/__init__.py`
- Create: `python/web/components/charts.py`
- Create: `python/tests/__init__.py`
- Create: `python/setup.py`
- Create: `requirements.txt`
- Create: `scripts/build.sh`
- Create: `scripts/run_tests.sh`
- Create: `docs/architecture.md`
- Create: `docs/formulas.md`
- Create: `docs/usage.md`
- Create: `docs/calibration_guide.md`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: 创建项目根目录和顶层 CMakeLists.txt**

```bash
mkdir -p cpp/{include,src,bindings,tests} python/{core,data/calibration_data,web/{pages,components},tests} docs scripts
```

`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(ModelCompute VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(cpp)
```

- [ ] **Step 2: 创建 C++ 子目录 CMakeLists.txt**

`cpp/CMakeLists.txt`:
```cmake
# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Core library
add_library(model_compute_core STATIC
  src/estimation_engine.cpp
  src/hardware_matcher.cpp
  src/calibration.cpp
)
target_include_directories(model_compute_core PUBLIC include)

# Python bindings
pybind11_add_module(model_compute bindings/pybind_module.cpp)
target_link_libraries(model_compute PRIVATE model_compute_core)

# Tests
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE model_compute_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_engine)
```

- [ ] **Step 3: 创建所有空的头文件和源文件骨架**

`cpp/include/estimation_engine.h`:
```cpp
#pragma once
#include <string>
#include <vector>

namespace model_compute {

enum class ModelType { DENSE, MOE, O1_REASONING, MULTIMODAL };
enum class Quantization { FP16, INT8, INT4 };

struct ModelParams {
    ModelType type;
    double param_billions;       // 参数量 (B)
    Quantization quant;
    int concurrency;             // 并发量
    int max_tokens;              // 最大 tokens 数
    // MoE 专用
    int num_experts = 0;         // 总专家数
    int active_experts = 0;      // 每 token 激活的专家数
    // o1 专用
    int reasoning_depth = 0;     // 0=无, 1=轻度, 2=中度, 3=重度
    // 多模态专用
    int image_resolution = 0;    // 图像分辨率 (px), 0=非多模态
    int num_images = 1;          // 图像数量
};

struct EstimationResult {
    double memory_gb;            // 显存需求 (GB)
    double flops_total;          // 总 FLOPs
    double bandwidth_gbs;        // 带宽需求 (GB/s)
    double kv_cache_gb;          // KV Cache 显存 (GB)
    double weight_memory_gb;     // 模型权重显存 (GB)
};

class EstimationEngine {
public:
    EstimationResult estimate(const ModelParams& params);

private:
    double bytes_per_param(Quantization q);
    double estimate_dense(const ModelParams& p, EstimationResult& r);
    double estimate_moe(const ModelParams& p, EstimationResult& r);
    double estimate_o1(const ModelParams& p, EstimationResult& r);
    double estimate_multimodal(const ModelParams& p, EstimationResult& r);
};

} // namespace model_compute
```

`cpp/include/hardware_matcher.h`:
```cpp
#pragma once
#include "estimation_engine.h"
#include <string>
#include <vector>

namespace model_compute {

struct HardwareSpec {
    std::string name;
    std::string vendor;
    std::string architecture;
    std::string type;  // "GPU" or "NPU"
    double fp16_tflops;
    double int8_tops;
    double fp32_tflops;
    double memory_gb;
    std::string memory_type;
    double memory_bandwidth_gbs;
    double nvlink_bandwidth_gbs;
    std::string pcie_version;
    double max_tdp_watts;
    double cost_per_unit;  // 参考价格
};

struct HardwareConfig {
    HardwareSpec hardware;
    int num_cards;
    double estimated_throughput;  // tokens/s
    double estimated_latency_ms;  // ms/token
    std::string bottleneck_type;  // "compute" or "memory"
    std::string parallel_strategy; // "TP", "PP", "TP+PP", "EP"
    bool meets_baseline;          // 是否满足 >= 10 tokens/s
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
```

`cpp/include/calibration.h`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <map>

namespace model_compute {

struct CalibrationPoint {
    std::string model_type;   // "dense", "moe", "o1", "multimodal"
    std::string hardware_name;
    double predicted_throughput;
    double actual_throughput;
    double predicted_memory;
    double actual_memory;
};

struct CalibrationFactor {
    double throughput_factor;  // actual / predicted
    double memory_factor;      // actual / predicted
    int num_points;
};

class Calibration {
public:
    void add_point(const CalibrationPoint& point);
    CalibrationFactor get_factor(const std::string& model_type, const std::string& hardware_name) const;
    double adjust_throughput(double predicted, const std::string& model_type, const std::string& hardware_name) const;
    double adjust_memory(double predicted, const std::string& model_type, const std::string& hardware_name) const;
    void load_from_file(const std::string& path);
    void save_to_file(const std::string& path) const;

private:
    std::map<std::pair<std::string, std::string>, std::vector<CalibrationPoint>> points_;
    std::map<std::pair<std::string, std::string>, CalibrationFactor> factors_;
    void recompute_factor(const std::string& model_type, const std::string& hardware_name);
};

} // namespace model_compute
```

- [ ] **Step 4: 创建 Python 包初始化文件和 requirements.txt**

`requirements.txt`:
```
streamlit>=1.30.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
pytest>=7.4.0
pybind11>=2.12.0
cmake>=3.28.0
```

`python/setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="model-compute",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.10",
)
```

- [ ] **Step 5: 创建 .gitignore 和 README.md**

`.gitignore`:
```
build/
__pycache__/
*.pyc
*.egg-info/
dist/
.venv/
.superpowers/
*.so
*.dylib
*.dll
```

`README.md`:
```markdown
# 模型对算力等效建模评估工具

## 简介
基于异构算力资源池，支持多负载的算力需求表征框架和动态调度工具。

## 支持的模型类型
- 稠密模型 (Dense): LLaMA, GPT, Qwen 等
- MoE 模型: Mixtral, DeepSeek-V2 等
- 类o1推理模型: DeepSeek-R1 等
- 多模态模型: LLaVA, Qwen-VL 等

## 安装
```bash
# 编译 C++ 模块
bash scripts/build.sh

# 安装 Python 依赖
pip install -r requirements.txt
pip install -e python/
```

## 使用
```bash
streamlit run python/web/app.py
```

## 测试
```bash
bash scripts/run_tests.sh
```
```

- [ ] **Step 6: 创建构建和测试脚本**

`scripts/build.sh`:
```bash
#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)
echo "Build complete. Python module at: build/cpp/"
```

`scripts/run_tests.sh`:
```bash
#!/bin/bash
set -e
echo "=== Running C++ tests ==="
cd build && ctest --output-on-failure
cd ..
echo "=== Running Python tests ==="
python -m pytest python/tests/ -v
echo "=== All tests passed ==="
```

- [ ] **Step 7: 验证构建**

```bash
bash scripts/build.sh
```

Expected: 构建成功，无编译错误。

- [ ] **Step 8: 提交初始脚手架**

```bash
git init
git add -A
git commit -m "feat: initial project scaffolding with C++/Python structure"
```

---

## Task 2: C++ 数据结构与基础估算 — Dense 模型

**Files:**
- Modify: `cpp/include/estimation_engine.h`
- Modify: `cpp/src/estimation_engine.cpp`
- Modify: `cpp/tests/test_engine.cpp`

- [ ] **Step 1: 编写 Dense 模型估算的失败测试**

`cpp/tests/test_engine.cpp`:
```cpp
#include <gtest/gtest.h>
#include "estimation_engine.h"
#include <cmath>

using namespace model_compute;

TEST(DenseEstimation, WeightMemoryFP16) {
    EstimationEngine engine;
    ModelParams params;
    params.type = ModelType::DENSE;
    params.param_billions = 7.0;  // 7B model
    params.quant = Quantization::FP16;
    params.concurrency = 1;
    params.max_tokens = 2048;

    auto result = engine.estimate(params);
    // 7B params * 2 bytes = 14 GB
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
    // 7B params * 1 byte = 7 GB
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
    // 70B params * 0.5 bytes = 35 GB
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
    // Total memory should be > weight memory (includes KV cache + overhead)
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
    // Higher concurrency should increase KV cache
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
    // FLOPs should be positive and reasonable
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
    // 70B * 2 bytes = 140 GB for weights alone
    EXPECT_NEAR(result.weight_memory_gb, 140.0, 2.0);
    EXPECT_GT(result.memory_gb, 140.0);
}
```

- [ ] **Step 2: 编译并运行测试，确认全部失败**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine
```

Expected: 编译错误（`estimate` 未实现）或链接错误。

- [ ] **Step 3: 实现 Dense 模型估算引擎**

`cpp/src/estimation_engine.cpp`:
```cpp
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

// LLaMA-style architecture: hidden_dim, num_layers derived from param count
// Reference: LLaMA-2 7B: 32 layers, 4096 hidden, 32 heads
//            LLaMA-2 13B: 40 layers, 5120 hidden, 40 heads
//            LLaMA-2 70B: 80 layers, 8192 hidden, 64 heads
static void infer_architecture(double param_b, int& num_layers, int& hidden_dim) {
    if (param_b <= 1.5)      { num_layers = 22;  hidden_dim = 2048; }
    else if (param_b <= 3.5) { num_layers = 26;  hidden_dim = 3200; }
    else if (param_b <= 8.0) { num_layers = 32;  hidden_dim = 4096; }
    else if (param_b <= 15.0){ num_layers = 40;  hidden_dim = 5120; }
    else if (param_b <= 35.0){ num_layers = 60;  hidden_dim = 6656; }
    else if (param_b <= 75.0){ num_layers = 80;  hidden_dim = 8192; }
    else                     { num_layers = 96;  hidden_dim = 12288; }
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
    }
    return result;
}

double EstimationEngine::estimate_dense(const ModelParams& p, EstimationResult& r) {
    double bpp = bytes_per_param(p.quant);
    double params_bytes = p.param_billions * 1e9 * bpp;
    r.weight_memory_gb = params_bytes / (1024.0 * 1024.0 * 1024.0);

    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);

    // KV Cache: 2 (K+V) * layers * hidden * seq_len * concurrency * bytes
    // For decode, seq_len grows incrementally. Use max_tokens as worst case.
    double kv_bytes = 2.0 * num_layers * hidden_dim * p.max_tokens * p.concurrency * bpp;
    r.kv_cache_gb = kv_bytes / (1024.0 * 1024.0 * 1024.0);

    // Activation memory (rough estimate)
    double activation_ratio = 0.02 * std::min(static_cast<double>(p.concurrency), 32.0);
    double activation_bytes = p.param_billions * 1e9 * activation_ratio * bpp;

    // Total with 10% overhead
    double total_bytes = params_bytes + kv_bytes + activation_bytes;
    r.memory_gb = total_bytes * 1.10 / (1024.0 * 1024.0 * 1024.0);

    // FLOPs: Prefill + Decode
    // Prefill: 2 * params * input_seq_len (assume input = max_tokens / 2)
    int input_seq = p.max_tokens / 2;
    int output_seq = p.max_tokens / 2;
    double prefill_flops = 2.0 * p.param_billions * 1e9 * input_seq;
    double decode_flops = 2.0 * p.param_billions * 1e9 * output_seq;
    r.flops_total = (prefill_flops + decode_flops) * p.concurrency;

    // Bandwidth estimate (for memory-bound decode phase)
    double bytes_per_token = p.param_billions * 1e9 * bpp + 2.0 * num_layers * hidden_dim * bpp;
    r.bandwidth_gbs = bytes_per_token * 10.0 / (1024.0 * 1024.0 * 1024.0); // 10 tokens/s baseline

    return r.memory_gb;
}

double EstimationEngine::estimate_moe(const ModelParams& p, EstimationResult& r) {
    // MoE: memory based on total params, FLOPs based on active params
    double bpp = bytes_per_param(p.quant);

    int num_experts = p.num_experts > 0 ? p.num_experts : 8;
    int active_experts = p.active_experts > 0 ? p.active_experts : 2;

    // Total params stored in memory (all experts)
    double total_params_bytes = p.param_billions * 1e9 * bpp;
    r.weight_memory_gb = total_params_bytes / (1024.0 * 1024.0 * 1024.0);

    // Active params for computation
    double active_ratio = static_cast<double>(active_experts) / num_experts;
    double active_params_b = p.param_billions * active_ratio;

    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);

    // KV Cache (shared attention layers)
    double shared_params_ratio = 0.3; // ~30% shared (attention + router)
    double kv_bytes = 2.0 * num_layers * hidden_dim * p.max_tokens * p.concurrency * bpp;
    r.kv_cache_gb = kv_bytes / (1024.0 * 1024.0 * 1024.0);

    double total_bytes = total_params_bytes + kv_bytes;
    r.memory_gb = total_bytes * 1.10 / (1024.0 * 1024.0 * 1024.0);

    // FLOPs based on active params + routing overhead
    int input_seq = p.max_tokens / 2;
    int output_seq = p.max_tokens / 2;
    double base_flops = 2.0 * active_params_b * 1e9 * (input_seq + output_seq);
    double routing_overhead = base_flops * 0.01;
    r.flops_total = (base_flops + routing_overhead) * p.concurrency;

    // Bandwidth: must load all expert weights
    double bytes_per_token = p.param_billions * 1e9 * bpp + 2.0 * num_layers * hidden_dim * bpp;
    r.bandwidth_gbs = bytes_per_token * 10.0 / (1024.0 * 1024.0 * 1024.0);

    return r.memory_gb;
}

double EstimationEngine::estimate_o1(const ModelParams& p, EstimationResult& r) {
    // o1 reasoning: extend Dense with reasoning token multiplier
    ModelParams dense_params = p;
    dense_params.type = ModelType::DENSE;

    // Reasoning depth multiplier
    double reasoning_multiplier = 1.0;
    switch (p.reasoning_depth) {
        case 1: reasoning_multiplier = 2.5; break;  // 轻度
        case 2: reasoning_multiplier = 4.0; break;  // 中度
        case 3: reasoning_multiplier = 7.5; break;  // 重度
        default: reasoning_multiplier = 1.0; break;
    }

    // Adjust max_tokens to include reasoning tokens
    dense_params.max_tokens = static_cast<int>(p.max_tokens * (1.0 + reasoning_multiplier));
    estimate_dense(dense_params, r);

    return r.memory_gb;
}

double EstimationEngine::estimate_multimodal(const ModelParams& p, EstimationResult& r) {
    // Multimodal: Dense backbone + Vision encoder
    ModelParams dense_params = p;
    dense_params.type = ModelType::DENSE;
    estimate_dense(dense_params, r);

    // Vision encoder overhead (ViT-L/14 ≈ 304M params, InternViT ≈ 6B)
    // Default to ViT-L/14 scale
    double vit_params_b = 0.304;
    if (p.param_billions > 30.0) vit_params_b = 6.0; // Use larger ViT for large models

    double bpp = bytes_per_param(p.quant);
    double vit_memory = vit_params_b * 1e9 * bpp / (1024.0 * 1024.0 * 1024.0);

    // Image tokens: (resolution / patch_size)^2, patch_size = 14
    int img_tokens = 0;
    if (p.image_resolution > 0) {
        int patches = p.image_resolution / 14;
        img_tokens = patches * patches;
    }

    // Additional KV cache for image tokens
    int num_layers, hidden_dim;
    infer_architecture(p.param_billions, num_layers, hidden_dim);
    double img_kv_bytes = 2.0 * num_layers * hidden_dim * img_tokens * p.num_images * p.concurrency * bpp;
    double img_kv_gb = img_kv_bytes / (1024.0 * 1024.0 * 1024.0);

    // Vision encoder FLOPs
    double vit_flops = 2.0 * vit_params_b * 1e9 * img_tokens * p.num_images * p.concurrency;

    r.weight_memory_gb += vit_memory;
    r.kv_cache_gb += img_kv_gb;
    r.memory_gb += vit_memory + img_kv_gb;
    r.flops_total += vit_flops;

    return r.memory_gb;
}

} // namespace model_compute
```

- [ ] **Step 4: 编译并运行测试**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine
```

Expected: 全部 7 个测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add cpp/include/estimation_engine.h cpp/src/estimation_engine.cpp cpp/tests/test_engine.cpp
git commit -m "feat: implement Dense/MoE/o1/multimodal estimation engine with tests"
```

---

## Task 3: C++ 硬件匹配器

**Files:**
- Modify: `cpp/src/hardware_matcher.cpp`
- Modify: `cpp/tests/test_engine.cpp`

- [ ] **Step 1: 编写硬件匹配器的失败测试**

在 `cpp/tests/test_engine.cpp` 末尾追加：

```cpp
#include "hardware_matcher.h"

TEST(HardwareMatcher, SingleCardSufficient) {
    HardwareMatcher matcher;
    EstimationResult est;
    est.memory_gb = 10.0;
    est.flops_total = 1e18;  // 1 EFLOPs
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
    est.memory_gb = 150.0;  // Needs more than single card
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
```

- [ ] **Step 2: 编译运行，确认测试失败**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine --gtest_filter="HardwareMatcher*"
```

Expected: 链接错误（`HardwareMatcher::match` 未实现）。

- [ ] **Step 3: 实现硬件匹配器**

`cpp/src/hardware_matcher.cpp`:
```cpp
#include "hardware_matcher.h"
#include <cmath>
#include <algorithm>

namespace model_compute {

int HardwareMatcher::calculate_cards_by_memory(double memory_needed, double card_memory) {
    return static_cast<int>(std::ceil(memory_needed / card_memory));
}

int HardwareMatcher::calculate_cards_by_compute(double flops_needed, double card_tflops) {
    // flops_needed is total FLOPs, card_tflops is TFLOPS
    // Assume we want to finish in ~1 second for throughput calc
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
    // TP communication: AllReduce per layer
    // Overhead ratio increases with more cards
    double tp_overhead = 0.0;
    if (hw.nvlink_bandwidth_gbs > 0) {
        tp_overhead = 0.05 * (cards - 1) / cards;  // ~5% per additional card
    } else {
        tp_overhead = 0.10 * (cards - 1) / cards;  // PCIe is slower
    }
    return std::min(tp_overhead, 0.30);  // Cap at 30%
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

        // Calculate required cards
        int cards_mem = calculate_cards_by_memory(estimation.memory_gb, hw.memory_gb);
        int cards_compute = 1;  // Most single cards can handle the compute
        config.num_cards = std::max(cards_mem, cards_compute);

        // Parallel strategy
        config.parallel_strategy = select_parallel_strategy(config.num_cards, model_params.type);

        // Communication overhead
        double comm_overhead = estimate_comm_overhead(config.num_cards, hw, model_params);

        // Throughput estimation (decode phase, memory-bound)
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
            config.estimated_throughput = 100.0;  // Default for small models
        }

        // Check compute bound
        double compute_throughput = hw.fp16_tflops * 1e12 / (2.0 * model_params.param_billions * 1e9);
        config.bottleneck_type = (config.estimated_throughput < compute_throughput) ? "memory" : "compute";
        config.estimated_throughput = std::min(config.estimated_throughput, compute_throughput);

        // Latency
        config.estimated_latency_ms = 1000.0 / config.estimated_throughput;

        // Baseline check
        config.meets_baseline = config.estimated_throughput >= baseline_throughput;

        results.push_back(config);
    }

    // Sort by throughput (descending)
    std::sort(results.begin(), results.end(),
        [](const HardwareConfig& a, const HardwareConfig& b) {
            return a.estimated_throughput > b.estimated_throughput;
        });

    return results;
}

} // namespace model_compute
```

- [ ] **Step 4: 编译运行测试**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine
```

Expected: 全部测试 PASS（包括之前的 7 个 + 新增 3 个）。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/hardware_matcher.cpp cpp/tests/test_engine.cpp
git commit -m "feat: implement hardware matcher with parallel strategy selection"
```

---

## Task 4: C++ 校准模块

**Files:**
- Modify: `cpp/src/calibration.cpp`
- Modify: `cpp/tests/test_engine.cpp`

- [ ] **Step 1: 编写校准模块的失败测试**

在 `cpp/tests/test_engine.cpp` 末尾追加：

```cpp
#include "calibration.h"

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

    std::string path = "/tmp/test_calibration.json";
    cal.save_to_file(path);

    Calibration cal2;
    cal2.load_from_file(path);
    auto factor = cal2.get_factor("dense", "A100 80G");
    EXPECT_NEAR(factor.throughput_factor, 0.8, 0.01);
}
```

- [ ] **Step 2: 编译运行，确认测试失败**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine --gtest_filter="Calibration*"
```

Expected: 链接错误。

- [ ] **Step 3: 实现校准模块**

`cpp/src/calibration.cpp`:
```cpp
#include "calibration.h"
#include <fstream>
#include <cmath>

namespace model_compute {

void Calibration::add_point(const CalibrationPoint& point) {
    auto key = std::make_pair(point.model_type, point.hardware_name);
    points_[key].push_back(point);
    recompute_factor(point.model_type, point.hardware_name);
}

CalibrationFactor Calibration::get_factor(const std::string& model_type, const std::string& hardware_name) const {
    auto key = std::make_pair(model_type, hardware_name);
    auto it = factors_.find(key);
    if (it != factors_.end()) {
        return it->second;
    }
    // Default: no correction
    CalibrationFactor default_factor;
    default_factor.throughput_factor = 1.0;
    default_factor.memory_factor = 1.0;
    default_factor.num_points = 0;
    return default_factor;
}

double Calibration::adjust_throughput(double predicted, const std::string& model_type, const std::string& hardware_name) const {
    return predicted * get_factor(model_type, hardware_name).throughput_factor;
}

double Calibration::adjust_memory(double predicted, const std::string& model_type, const std::string& hardware_name) const {
    return predicted * get_factor(model_type, hardware_name).memory_factor;
}

void Calibration::recompute_factor(const std::string& model_type, const std::string& hardware_name) {
    auto key = std::make_pair(model_type, hardware_name);
    auto& pts = points_[key];
    if (pts.empty()) return;

    double sum_tp_factor = 0.0;
    double sum_mem_factor = 0.0;
    for (const auto& p : pts) {
        sum_tp_factor += p.actual_throughput / p.predicted_throughput;
        sum_mem_factor += p.actual_memory / p.predicted_memory;
    }

    CalibrationFactor f;
    f.throughput_factor = sum_tp_factor / pts.size();
    f.memory_factor = sum_mem_factor / pts.size();
    f.num_points = static_cast<int>(pts.size());
    factors_[key] = f;
}

void Calibration::load_from_file(const std::string& path) {
    // Simple JSON parsing using ifstream (no external dependency)
    // Format: lines of "model_type,hardware_name,pred_tp,act_tp,pred_mem,act_mem"
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        CalibrationPoint pt;
        // Parse CSV-like format
        size_t pos = 0;
        auto next = [&]() -> std::string {
            size_t end = line.find(',', pos);
            std::string val = line.substr(pos, end - pos);
            pos = (end == std::string::npos) ? end : end + 1;
            return val;
        };
        pt.model_type = next();
        pt.hardware_name = next();
        pt.predicted_throughput = std::stod(next());
        pt.actual_throughput = std::stod(next());
        pt.predicted_memory = std::stod(next());
        pt.actual_memory = std::stod(next());
        add_point(pt);
    }
}

void Calibration::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    file << "# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem\n";
    for (const auto& [key, pts] : points_) {
        for (const auto& p : pts) {
            file << p.model_type << ","
                 << p.hardware_name << ","
                 << p.predicted_throughput << ","
                 << p.actual_throughput << ","
                 << p.predicted_memory << ","
                 << p.actual_memory << "\n";
        }
    }
}

} // namespace model_compute
```

- [ ] **Step 4: 编译运行测试**

```bash
cd build && cmake --build . --target test_engine && ./cpp/tests/test_engine
```

Expected: 全部 15 个测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add cpp/src/calibration.cpp cpp/tests/test_engine.cpp
git commit -m "feat: implement calibration module with save/load support"
```

---

## Task 5: pybind11 绑定层

**Files:**
- Modify: `cpp/bindings/pybind_module.cpp`

- [ ] **Step 1: 实现 pybind11 绑定**

`cpp/bindings/pybind_module.cpp`:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "estimation_engine.h"
#include "hardware_matcher.h"
#include "calibration.h"

namespace py = pybind11;
using namespace model_compute;

PYBIND11_MODULE(model_compute, m) {
    m.doc() = "Model Compute Estimation Engine";

    // Enums
    py::enum_<ModelType>(m, "ModelType")
        .value("DENSE", ModelType::DENSE)
        .value("MOE", ModelType::MOE)
        .value("O1_REASONING", ModelType::O1_REASONING)
        .value("MULTIMODAL", ModelType::MULTIMODAL);

    py::enum_<Quantization>(m, "Quantization")
        .value("FP16", Quantization::FP16)
        .value("INT8", Quantization::INT8)
        .value("INT4", Quantization::INT4);

    // ModelParams
    py::class_<ModelParams>(m, "ModelParams")
        .def(py::init<>())
        .def_readwrite("type", &ModelParams::type)
        .def_readwrite("param_billions", &ModelParams::param_billions)
        .def_readwrite("quant", &ModelParams::quant)
        .def_readwrite("concurrency", &ModelParams::concurrency)
        .def_readwrite("max_tokens", &ModelParams::max_tokens)
        .def_readwrite("num_experts", &ModelParams::num_experts)
        .def_readwrite("active_experts", &ModelParams::active_experts)
        .def_readwrite("reasoning_depth", &ModelParams::reasoning_depth)
        .def_readwrite("image_resolution", &ModelParams::image_resolution)
        .def_readwrite("num_images", &ModelParams::num_images);

    // EstimationResult
    py::class_<EstimationResult>(m, "EstimationResult")
        .def_readonly("memory_gb", &EstimationResult::memory_gb)
        .def_readonly("flops_total", &EstimationResult::flops_total)
        .def_readonly("bandwidth_gbs", &EstimationResult::bandwidth_gbs)
        .def_readonly("kv_cache_gb", &EstimationResult::kv_cache_gb)
        .def_readonly("weight_memory_gb", &EstimationResult::weight_memory_gb);

    // EstimationEngine
    py::class_<EstimationEngine>(m, "EstimationEngine")
        .def(py::init<>())
        .def("estimate", &EstimationEngine::estimate);

    // HardwareSpec
    py::class_<HardwareSpec>(m, "HardwareSpec")
        .def(py::init<>())
        .def_readwrite("name", &HardwareSpec::name)
        .def_readwrite("vendor", &HardwareSpec::vendor)
        .def_readwrite("architecture", &HardwareSpec::architecture)
        .def_readwrite("type", &HardwareSpec::type)
        .def_readwrite("fp16_tflops", &HardwareSpec::fp16_tflops)
        .def_readwrite("int8_tops", &HardwareSpec::int8_tops)
        .def_readwrite("fp32_tflops", &HardwareSpec::fp32_tflops)
        .def_readwrite("memory_gb", &HardwareSpec::memory_gb)
        .def_readwrite("memory_type", &HardwareSpec::memory_type)
        .def_readwrite("memory_bandwidth_gbs", &HardwareSpec::memory_bandwidth_gbs)
        .def_readwrite("nvlink_bandwidth_gbs", &HardwareSpec::nvlink_bandwidth_gbs)
        .def_readwrite("pcie_version", &HardwareSpec::pcie_version)
        .def_readwrite("max_tdp_watts", &HardwareSpec::max_tdp_watts)
        .def_readwrite("cost_per_unit", &HardwareSpec::cost_per_unit);

    // HardwareConfig
    py::class_<HardwareConfig>(m, "HardwareConfig")
        .def_readonly("hardware", &HardwareConfig::hardware)
        .def_readonly("num_cards", &HardwareConfig::num_cards)
        .def_readonly("estimated_throughput", &HardwareConfig::estimated_throughput)
        .def_readonly("estimated_latency_ms", &HardwareConfig::estimated_latency_ms)
        .def_readonly("bottleneck_type", &HardwareConfig::bottleneck_type)
        .def_readonly("parallel_strategy", &HardwareConfig::parallel_strategy)
        .def_readonly("meets_baseline", &HardwareConfig::meets_baseline);

    // HardwareMatcher
    py::class_<HardwareMatcher>(m, "HardwareMatcher")
        .def(py::init<>())
        .def("match", &HardwareMatcher::match,
             py::arg("estimation"), py::arg("model_params"),
             py::arg("hardware_pool"), py::arg("baseline_throughput") = 10.0);

    // Calibration
    py::class_<Calibration>(m, "Calibration")
        .def(py::init<>())
        .def("add_point", &Calibration::add_point)
        .def("get_factor", &Calibration::get_factor)
        .def("adjust_throughput", &Calibration::adjust_throughput)
        .def("adjust_memory", &Calibration::adjust_memory)
        .def("load_from_file", &Calibration::load_from_file)
        .def("save_to_file", &Calibration::save_to_file);

    // CalibrationPoint
    py::class_<CalibrationPoint>(m, "CalibrationPoint")
        .def(py::init<>())
        .def_readwrite("model_type", &CalibrationPoint::model_type)
        .def_readwrite("hardware_name", &CalibrationPoint::hardware_name)
        .def_readwrite("predicted_throughput", &CalibrationPoint::predicted_throughput)
        .def_readwrite("actual_throughput", &CalibrationPoint::actual_throughput)
        .def_readwrite("predicted_memory", &CalibrationPoint::predicted_memory)
        .def_readwrite("actual_memory", &CalibrationPoint::actual_memory);

    // CalibrationFactor
    py::class_<CalibrationFactor>(m, "CalibrationFactor")
        .def_readonly("throughput_factor", &CalibrationFactor::throughput_factor)
        .def_readonly("memory_factor", &CalibrationFactor::memory_factor)
        .def_readonly("num_points", &CalibrationFactor::num_points);
}
```

- [ ] **Step 2: 编译 Python 绑定模块**

```bash
cd build && cmake --build . --target model_compute
```

Expected: 编译成功，生成 `model_compute.so`（或 `.pyd`）。

- [ ] **Step 3: 验证 Python 可以导入**

```bash
cd build && python -c "
import sys; sys.path.insert(0, 'cpp')
import model_compute
print('Module loaded:', dir(model_compute))
mp = model_compute.ModelParams()
mp.type = model_compute.ModelType.DENSE
mp.param_billions = 7.0
mp.quant = model_compute.Quantization.FP16
mp.concurrency = 1
mp.max_tokens = 2048
engine = model_compute.EstimationEngine()
result = engine.estimate(mp)
print(f'Memory: {result.memory_gb:.1f} GB')
print(f'FLOPs: {result.flops_total:.2e}')
print('SUCCESS')
"
```

Expected: 输出 Memory 和 FLOPs 数值，最后打印 SUCCESS。

- [ ] **Step 4: 提交**

```bash
git add cpp/bindings/pybind_module.cpp
git commit -m "feat: add pybind11 bindings for C++ estimation engine"
```

---

## Task 6: Python 业务层 — 硬件数据库

**Files:**
- Create: `python/data/hardware_specs.json`
- Modify: `python/core/hardware_db.py`
- Create: `python/tests/test_hardware_db.py`

- [ ] **Step 1: 创建硬件规格 JSON 数据文件**

`python/data/hardware_specs.json`:
```json
{
  "hardware": [
    {
      "name": "NVIDIA A100 40GB",
      "vendor": "NVIDIA",
      "architecture": "Ampere",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 312,
        "int8_tops": 624,
        "fp32_tflops": 19.5,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "memory_bandwidth_gbs": 1555,
        "nvlink_bandwidth_gbs": 600,
        "pcie_version": "4.0",
        "max_tdp_watts": 250
      },
      "cost_per_unit": 8000,
      "notes": "Ampere架构推理GPU"
    },
    {
      "name": "NVIDIA A100 80GB",
      "vendor": "NVIDIA",
      "architecture": "Ampere",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 312,
        "int8_tops": 624,
        "fp32_tflops": 19.5,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "memory_bandwidth_gbs": 2039,
        "nvlink_bandwidth_gbs": 600,
        "pcie_version": "4.0",
        "max_tdp_watts": 300
      },
      "cost_per_unit": 12000,
      "notes": "主流训练/推理GPU"
    },
    {
      "name": "NVIDIA H100 80GB",
      "vendor": "NVIDIA",
      "architecture": "Hopper",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 990,
        "int8_tops": 1979,
        "fp32_tflops": 67,
        "memory_gb": 80,
        "memory_type": "HBM3",
        "memory_bandwidth_gbs": 3350,
        "nvlink_bandwidth_gbs": 900,
        "pcie_version": "5.0",
        "max_tdp_watts": 700
      },
      "cost_per_unit": 25000,
      "notes": "旗舰级训练/推理GPU"
    },
    {
      "name": "NVIDIA H200",
      "vendor": "NVIDIA",
      "architecture": "Hopper",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 990,
        "int8_tops": 1979,
        "fp32_tflops": 67,
        "memory_gb": 141,
        "memory_type": "HBM3e",
        "memory_bandwidth_gbs": 4800,
        "nvlink_bandwidth_gbs": 900,
        "pcie_version": "5.0",
        "max_tdp_watts": 700
      },
      "cost_per_unit": 35000,
      "notes": "大显存Hopper GPU"
    },
    {
      "name": "NVIDIA L40S",
      "vendor": "NVIDIA",
      "architecture": "Ada Lovelace",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 362,
        "int8_tops": 733,
        "fp32_tflops": 91.6,
        "memory_gb": 48,
        "memory_type": "GDDR6X",
        "memory_bandwidth_gbs": 864,
        "nvlink_bandwidth_gbs": 0,
        "pcie_version": "4.0",
        "max_tdp_watts": 350
      },
      "cost_per_unit": 8000,
      "notes": "推理优化GPU，无NVLink"
    },
    {
      "name": "NVIDIA RTX 4090",
      "vendor": "NVIDIA",
      "architecture": "Ada Lovelace",
      "type": "GPU",
      "specs": {
        "fp16_tflops": 330,
        "int8_tops": 660,
        "fp32_tflops": 82.6,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "memory_bandwidth_gbs": 1008,
        "nvlink_bandwidth_gbs": 0,
        "pcie_version": "4.0",
        "max_tdp_watts": 450
      },
      "cost_per_unit": 1600,
      "notes": "消费级GPU"
    },
    {
      "name": "华为 Ascend 910B",
      "vendor": "华为",
      "architecture": "达芬奇",
      "type": "NPU",
      "specs": {
        "fp16_tflops": 320,
        "int8_tops": 640,
        "fp32_tflops": 0,
        "memory_gb": 64,
        "memory_type": "HBM2e",
        "memory_bandwidth_gbs": 1200,
        "nvlink_bandwidth_gbs": 0,
        "pcie_version": "4.0",
        "max_tdp_watts": 310
      },
      "cost_per_unit": 10000,
      "notes": "昇腾系列NPU，支持HCCS互联"
    },
    {
      "name": "华为 Ascend 300I Duo",
      "vendor": "华为",
      "architecture": "达芬奇",
      "type": "NPU",
      "specs": {
        "fp16_tflops": 256,
        "int8_tops": 512,
        "fp32_tflops": 0,
        "memory_gb": 64,
        "memory_type": "HBM2e",
        "memory_bandwidth_gbs": 800,
        "nvlink_bandwidth_gbs": 0,
        "pcie_version": "4.0",
        "max_tdp_watts": 200
      },
      "cost_per_unit": 6000,
      "notes": "推理专用NPU"
    },
    {
      "name": "寒武纪 MLU370",
      "vendor": "寒武纪",
      "architecture": "MLUarch03",
      "type": "NPU",
      "specs": {
        "fp16_tflops": 256,
        "int8_tops": 512,
        "fp32_tflops": 0,
        "memory_gb": 48,
        "memory_type": "HBM2e",
        "memory_bandwidth_gbs": 614,
        "nvlink_bandwidth_gbs": 0,
        "pcie_version": "4.0",
        "max_tdp_watts": 250
      },
      "cost_per_unit": 7000,
      "notes": "国产NPU"
    }
  ]
}
```

- [ ] **Step 2: 编写硬件数据库管理的失败测试**

`python/tests/test_hardware_db.py`:
```python
import pytest
import json
import tempfile
import os
from python.core.hardware_db import HardwareDB


def test_load_default_hardware():
    db = HardwareDB()
    hw_list = db.list_hardware()
    assert len(hw_list) >= 9  # At least 9 preloaded hardware
    names = [h["name"] for h in hw_list]
    assert "NVIDIA A100 80GB" in names
    assert "华为 Ascend 910B" in names


def test_get_hardware_by_name():
    db = HardwareDB()
    hw = db.get_hardware("NVIDIA A100 80GB")
    assert hw is not None
    assert hw["specs"]["memory_gb"] == 80
    assert hw["specs"]["fp16_tflops"] == 312


def test_get_hardware_not_found():
    db = HardwareDB()
    hw = db.get_hardware("NonExistent GPU")
    assert hw is None


def test_add_custom_hardware():
    db = HardwareDB()
    custom = {
        "name": "Custom GPU",
        "vendor": "Custom",
        "architecture": "Test",
        "type": "GPU",
        "specs": {
            "fp16_tflops": 500,
            "int8_tops": 1000,
            "fp32_tflops": 50,
            "memory_gb": 128,
            "memory_type": "HBM3",
            "memory_bandwidth_gbs": 3000,
            "nvlink_bandwidth_gbs": 800,
            "pcie_version": "5.0",
            "max_tdp_watts": 400
        },
        "cost_per_unit": 20000,
        "notes": "Test GPU"
    }
    db.add_hardware(custom)
    hw = db.get_hardware("Custom GPU")
    assert hw is not None
    assert hw["specs"]["memory_gb"] == 128


def test_remove_hardware():
    db = HardwareDB()
    db.add_hardware({
        "name": "Temp GPU",
        "vendor": "Temp",
        "architecture": "Temp",
        "type": "GPU",
        "specs": {
            "fp16_tflops": 100,
            "int8_tops": 200,
            "fp32_tflops": 10,
            "memory_gb": 16,
            "memory_type": "GDDR6",
            "memory_bandwidth_gbs": 500,
            "nvlink_bandwidth_gbs": 0,
            "pcie_version": "4.0",
            "max_tdp_watts": 150
        },
        "cost_per_unit": 500
    })
    assert db.get_hardware("Temp GPU") is not None
    db.remove_hardware("Temp GPU")
    assert db.get_hardware("Temp GPU") is None


def test_to_cpp_hardware_list():
    db = HardwareDB()
    cpp_list = db.to_cpp_hardware_list()
    assert len(cpp_list) >= 9
    # Check first item has correct structure
    assert hasattr(cpp_list[0], "name")
    assert hasattr(cpp_list[0], "fp16_tflops")


def test_save_and_load(tmp_path):
    db = HardwareDB()
    save_path = str(tmp_path / "test_hw.json")
    db.save(save_path)

    db2 = HardwareDB(save_path)
    hw = db2.get_hardware("NVIDIA A100 80GB")
    assert hw is not None
    assert hw["specs"]["memory_gb"] == 80
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
python -m pytest python/tests/test_hardware_db.py -v
```

Expected: ImportError（`hardware_db` 不存在）。

- [ ] **Step 4: 实现硬件数据库管理**

`python/core/hardware_db.py`:
```python
import json
import os
from pathlib import Path
from typing import Optional

# Import C++ bindings
import sys
_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as _mc
except ImportError:
    _mc = None


_DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "hardware_specs.json"


class HardwareDB:
    def __init__(self, data_path: Optional[str] = None):
        self._data_path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
        self._hardware: list[dict] = []
        self._load()

    def _load(self):
        if self._data_path.exists():
            with open(self._data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._hardware = data.get("hardware", [])

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else self._data_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"hardware": self._hardware}, f, ensure_ascii=False, indent=2)

    def list_hardware(self) -> list[dict]:
        return self._hardware

    def get_hardware(self, name: str) -> Optional[dict]:
        for hw in self._hardware:
            if hw["name"] == name:
                return hw
        return None

    def add_hardware(self, hw: dict):
        # Remove existing with same name
        self._hardware = [h for h in self._hardware if h["name"] != hw["name"]]
        self._hardware.append(hw)

    def remove_hardware(self, name: str):
        self._hardware = [h for h in self._hardware if h["name"] != name]

    def to_cpp_hardware_list(self) -> list:
        """Convert to C++ HardwareSpec objects for the engine."""
        if _mc is None:
            raise RuntimeError("C++ module not available. Build with: bash scripts/build.sh")

        result = []
        for hw in self._hardware:
            spec = _mc.HardwareSpec()
            spec.name = hw["name"]
            spec.vendor = hw.get("vendor", "")
            spec.architecture = hw.get("architecture", "")
            spec.type = hw.get("type", "GPU")
            s = hw.get("specs", {})
            spec.fp16_tflops = s.get("fp16_tflops", 0)
            spec.int8_tops = s.get("int8_tops", 0)
            spec.fp32_tflops = s.get("fp32_tflops", 0)
            spec.memory_gb = s.get("memory_gb", 0)
            spec.memory_type = s.get("memory_type", "")
            spec.memory_bandwidth_gbs = s.get("memory_bandwidth_gbs", 0)
            spec.nvlink_bandwidth_gbs = s.get("nvlink_bandwidth_gbs", 0)
            spec.pcie_version = s.get("pcie_version", "")
            spec.max_tdp_watts = s.get("max_tdp_watts", 0)
            spec.cost_per_unit = hw.get("cost_per_unit", 0)
            result.append(spec)
        return result
```

- [ ] **Step 5: 运行测试**

```bash
python -m pytest python/tests/test_hardware_db.py -v
```

Expected: 全部 7 个测试 PASS。

- [ ] **Step 6: 提交**

```bash
git add python/data/hardware_specs.json python/core/hardware_db.py python/tests/test_hardware_db.py
git commit -m "feat: implement hardware database with JSON storage and C++ bridge"
```

---

## Task 7: Python 业务层 — 模型分析器

**Files:**
- Create: `python/data/model_presets.json`
- Modify: `python/core/model_analyzer.py`
- Create: `python/tests/test_model_analyzer.py`

- [ ] **Step 1: 创建模型预设数据**

`python/data/model_presets.json`:
```json
{
  "presets": {
    "dense": [
      {"name": "LLaMA-2 7B", "param_billions": 7.0, "num_layers": 32, "hidden_dim": 4096},
      {"name": "LLaMA-2 13B", "param_billions": 13.0, "num_layers": 40, "hidden_dim": 5120},
      {"name": "LLaMA-2 70B", "param_billions": 70.0, "num_layers": 80, "hidden_dim": 8192},
      {"name": "Qwen-7B", "param_billions": 7.0, "num_layers": 32, "hidden_dim": 4096},
      {"name": "Qwen-14B", "param_billions": 14.0, "num_layers": 40, "hidden_dim": 5120},
      {"name": "Qwen-72B", "param_billions": 72.0, "num_layers": 80, "hidden_dim": 8192}
    ],
    "moe": [
      {"name": "Mixtral 8x7B", "param_billions": 46.7, "num_experts": 8, "active_experts": 2, "num_layers": 32, "hidden_dim": 4096},
      {"name": "Mixtral 8x22B", "param_billions": 141.0, "num_experts": 8, "active_experts": 2, "num_layers": 56, "hidden_dim": 6144},
      {"name": "DeepSeek-V2", "param_billions": 236.0, "num_experts": 160, "active_experts": 6, "num_layers": 60, "hidden_dim": 5120}
    ],
    "o1_reasoning": [
      {"name": "DeepSeek-R1", "param_billions": 671.0, "reasoning_depth": 2, "num_layers": 61, "hidden_dim": 7168},
      {"name": "QwQ-32B", "param_billions": 32.0, "reasoning_depth": 2, "num_layers": 64, "hidden_dim": 5120}
    ],
    "multimodal": [
      {"name": "LLaVA-1.5 7B", "param_billions": 7.0, "vit_params_billions": 0.304, "image_resolution": 336, "num_layers": 32, "hidden_dim": 4096},
      {"name": "LLaVA-1.5 13B", "param_billions": 13.0, "vit_params_billions": 0.304, "image_resolution": 336, "num_layers": 40, "hidden_dim": 5120},
      {"name": "Qwen-VL-Chat", "param_billions": 9.6, "vit_params_billions": 0.675, "image_resolution": 448, "num_layers": 32, "hidden_dim": 4096}
    ]
  }
}
```

- [ ] **Step 2: 编写模型分析器的失败测试**

`python/tests/test_model_analyzer.py`:
```python
import pytest
from python.core.model_analyzer import ModelAnalyzer


def test_load_presets():
    analyzer = ModelAnalyzer()
    presets = analyzer.list_presets()
    assert "dense" in presets
    assert "moe" in presets
    assert len(presets["dense"]) >= 6


def test_get_preset():
    analyzer = ModelAnalyzer()
    preset = analyzer.get_preset("dense", "LLaMA-2 7B")
    assert preset is not None
    assert preset["param_billions"] == 7.0


def test_create_params_from_preset():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="dense",
        preset_name="LLaMA-2 7B",
        quant="FP16",
        concurrency=1,
        max_tokens=2048
    )
    assert params.type.name == "DENSE"
    assert params.param_billions == 7.0
    assert params.concurrency == 1


def test_create_params_custom():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="dense",
        param_billions=13.0,
        quant="INT8",
        concurrency=16,
        max_tokens=4096
    )
    assert params.param_billions == 13.0
    assert params.concurrency == 16


def test_create_moe_params():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="moe",
        preset_name="Mixtral 8x7B",
        quant="FP16",
        concurrency=1,
        max_tokens=2048
    )
    assert params.num_experts == 8
    assert params.active_experts == 2


def test_create_o1_params():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="o1_reasoning",
        preset_name="DeepSeek-R1",
        quant="FP16",
        concurrency=1,
        max_tokens=4096,
        reasoning_depth=2
    )
    assert params.reasoning_depth == 2


def test_create_multimodal_params():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="multimodal",
        preset_name="LLaVA-1.5 7B",
        quant="FP16",
        concurrency=1,
        max_tokens=2048,
        image_resolution=336,
        num_images=1
    )
    assert params.image_resolution == 336
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
python -m pytest python/tests/test_model_analyzer.py -v
```

Expected: ImportError。

- [ ] **Step 4: 实现模型分析器**

`python/core/model_analyzer.py`:
```python
import json
import sys
from pathlib import Path
from typing import Optional

_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as _mc
except ImportError:
    _mc = None

_PRESETS_PATH = Path(__file__).parent.parent / "data" / "model_presets.json"


_QUANT_MAP = {
    "FP16": _mc.Quantization.FP16 if _mc else None,
    "INT8": _mc.Quantization.INT8 if _mc else None,
    "INT4": _mc.Quantization.INT4 if _mc else None,
}

_TYPE_MAP = {
    "dense": _mc.ModelType.DENSE if _mc else None,
    "moe": _mc.ModelType.MOE if _mc else None,
    "o1_reasoning": _mc.ModelType.O1_REASONING if _mc else None,
    "multimodal": _mc.ModelType.MULTIMODAL if _mc else None,
}


class ModelAnalyzer:
    def __init__(self, presets_path: Optional[str] = None):
        self._path = Path(presets_path) if presets_path else _PRESETS_PATH
        self._presets: dict = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._presets = json.load(f).get("presets", {})

    def list_presets(self) -> dict:
        return {k: [p["name"] for p in v] for k, v in self._presets.items()}

    def get_preset(self, model_type: str, preset_name: str) -> Optional[dict]:
        for p in self._presets.get(model_type, []):
            if p["name"] == preset_name:
                return p
        return None

    def create_params(
        self,
        model_type: str,
        preset_name: Optional[str] = None,
        param_billions: Optional[float] = None,
        quant: str = "FP16",
        concurrency: int = 1,
        max_tokens: int = 2048,
        num_experts: int = 0,
        active_experts: int = 0,
        reasoning_depth: int = 0,
        image_resolution: int = 0,
        num_images: int = 1,
    ):
        if _mc is None:
            raise RuntimeError("C++ module not available. Build with: bash scripts/build.sh")

        params = _mc.ModelParams()
        params.type = _TYPE_MAP[model_type]
        params.quant = _QUANT_MAP[quant]
        params.concurrency = concurrency
        params.max_tokens = max_tokens

        preset = self.get_preset(model_type, preset_name) if preset_name else None

        if preset:
            params.param_billions = preset.get("param_billions", 0)
            params.num_experts = preset.get("num_experts", 0)
            params.active_experts = preset.get("active_experts", 0)
            params.reasoning_depth = preset.get("reasoning_depth", 0)
            params.image_resolution = preset.get("image_resolution", 0)
        elif param_billions is not None:
            params.param_billions = param_billions
            params.num_experts = num_experts
            params.active_experts = active_experts
            params.reasoning_depth = reasoning_depth
            params.image_resolution = image_resolution

        # Override reasoning_depth and multimodal params if explicitly provided
        if reasoning_depth > 0:
            params.reasoning_depth = reasoning_depth
        if image_resolution > 0:
            params.image_resolution = image_resolution
        params.num_images = num_images

        return params
```

- [ ] **Step 5: 运行测试**

```bash
python -m pytest python/tests/test_model_analyzer.py -v
```

Expected: 全部 7 个测试 PASS。

- [ ] **Step 6: 提交**

```bash
git add python/data/model_presets.json python/core/model_analyzer.py python/tests/test_model_analyzer.py
git commit -m "feat: implement model analyzer with presets support"
```

---

## Task 8: Python 业务层 — 校准管理器

**Files:**
- Modify: `python/core/calibration_mgr.py`
- Create: `python/tests/test_calibration_mgr.py`

- [ ] **Step 1: 编写校准管理器的失败测试**

`python/tests/test_calibration_mgr.py`:
```python
import pytest
import tempfile
import os
from python.core.calibration_mgr import CalibrationManager


def test_add_calibration_point():
    mgr = CalibrationManager()
    mgr.add_point(
        model_type="dense",
        hardware_name="A100 80G",
        predicted_throughput=20.0,
        actual_throughput=16.0,
        predicted_memory=14.0,
        actual_memory=15.5
    )
    factor = mgr.get_factor("dense", "A100 80G")
    assert abs(factor.throughput_factor - 0.8) < 0.01
    assert abs(factor.memory_factor - 1.107) < 0.01


def test_default_factor():
    mgr = CalibrationManager()
    factor = mgr.get_factor("moe", "H100")
    assert abs(factor.throughput_factor - 1.0) < 0.001
    assert abs(factor.memory_factor - 1.0) < 0.001


def test_import_csv(tmp_path):
    csv_content = "dense,A100,20.0,16.0,14.0,15.0\ndense,A100,30.0,25.5,28.0,30.0\n"
    csv_path = str(tmp_path / "cal.csv")
    with open(csv_path, "w") as f:
        f.write(csv_content)

    mgr = CalibrationManager()
    count = mgr.import_csv(csv_path)
    assert count == 2
    factor = mgr.get_factor("dense", "A100")
    assert factor.num_points == 2


def test_save_and_load(tmp_path):
    mgr = CalibrationManager()
    mgr.add_point("dense", "A100", 20.0, 16.0, 14.0, 15.0)
    save_path = str(tmp_path / "cal.csv")
    mgr.save(save_path)

    mgr2 = CalibrationManager()
    mgr2.load(save_path)
    factor = mgr2.get_factor("dense", "A100")
    assert abs(factor.throughput_factor - 0.8) < 0.01


def test_list_calibrations():
    mgr = CalibrationManager()
    mgr.add_point("dense", "A100", 20.0, 16.0, 14.0, 15.0)
    mgr.add_point("moe", "H100", 30.0, 27.0, 50.0, 55.0)
    entries = mgr.list_entries()
    assert len(entries) == 2
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest python/tests/test_calibration_mgr.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现校准管理器**

`python/core/calibration_mgr.py`:
```python
import sys
from pathlib import Path
from typing import Optional

_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as _mc
except ImportError:
    _mc = None

_DEFAULT_CAL_DIR = Path(__file__).parent.parent / "data" / "calibration_data"


class CalibrationManager:
    def __init__(self, cal_dir: Optional[str] = None):
        self._dir = Path(cal_dir) if cal_dir else _DEFAULT_CAL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        if _mc:
            self._cal = _mc.Calibration()
        else:
            self._cal = None
        self._points: list[dict] = []

    def add_point(
        self,
        model_type: str,
        hardware_name: str,
        predicted_throughput: float,
        actual_throughput: float,
        predicted_memory: float,
        actual_memory: float,
    ):
        pt_dict = {
            "model_type": model_type,
            "hardware_name": hardware_name,
            "predicted_throughput": predicted_throughput,
            "actual_throughput": actual_throughput,
            "predicted_memory": predicted_memory,
            "actual_memory": actual_memory,
        }
        self._points.append(pt_dict)

        if self._cal:
            pt = _mc.CalibrationPoint()
            pt.model_type = model_type
            pt.hardware_name = hardware_name
            pt.predicted_throughput = predicted_throughput
            pt.actual_throughput = actual_throughput
            pt.predicted_memory = predicted_memory
            pt.actual_memory = actual_memory
            self._cal.add_point(pt)

    def get_factor(self, model_type: str, hardware_name: str):
        if self._cal:
            return self._cal.get_factor(model_type, hardware_name)

        # Fallback: compute from stored points
        matching = [
            p for p in self._points
            if p["model_type"] == model_type and p["hardware_name"] == hardware_name
        ]
        if not matching:
            return type("Factor", (), {"throughput_factor": 1.0, "memory_factor": 1.0, "num_points": 0})()

        avg_tp = sum(p["actual_throughput"] / p["predicted_throughput"] for p in matching) / len(matching)
        avg_mem = sum(p["actual_memory"] / p["predicted_memory"] for p in matching) / len(matching)
        return type("Factor", (), {
            "throughput_factor": avg_tp,
            "memory_factor": avg_mem,
            "num_points": len(matching),
        })()

    def import_csv(self, path: str) -> int:
        count = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 6:
                    self.add_point(
                        model_type=parts[0].strip(),
                        hardware_name=parts[1].strip(),
                        predicted_throughput=float(parts[2]),
                        actual_throughput=float(parts[3]),
                        predicted_memory=float(parts[4]),
                        actual_memory=float(parts[5]),
                    )
                    count += 1
        return count

    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else self._dir / "calibration.csv"
        with open(save_path, "w") as f:
            f.write("# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem\n")
            for p in self._points:
                f.write(f"{p['model_type']},{p['hardware_name']},"
                        f"{p['predicted_throughput']},{p['actual_throughput']},"
                        f"{p['predicted_memory']},{p['actual_memory']}\n")

    def load(self, path: Optional[str] = None):
        load_path = Path(path) if path else self._dir / "calibration.csv"
        if load_path.exists():
            self.import_csv(str(load_path))

    def list_entries(self) -> list[dict]:
        return self._points
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest python/tests/test_calibration_mgr.py -v
```

Expected: 全部 5 个测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add python/core/calibration_mgr.py python/tests/test_calibration_mgr.py
git commit -m "feat: implement calibration manager with CSV import/export"
```

---

## Task 9: Streamlit Web UI — 算力估算主页面

**Files:**
- Modify: `python/web/app.py`
- Modify: `python/web/pages/estimation.py`
- Create: `python/web/components/charts.py`

- [ ] **Step 1: 创建图表组件**

`python/web/components/charts.py`:
```python
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render_throughput_bar(configs: list[dict]) -> go.Figure:
    """Render bar chart comparing throughput across hardware options."""
    df = pd.DataFrame(configs)
    fig = px.bar(
        df, x="hardware", y="throughput", color="meets_baseline",
        title="预估吞吐量对比 (tokens/s)",
        labels={"hardware": "硬件型号", "throughput": "吞吐量", "meets_baseline": "满足基线"},
        color_discrete_map={True: "#2ed573", False: "#ff4757"},
    )
    fig.add_hline(y=10, line_dash="dash", line_color="yellow", annotation_text="基线 10 tokens/s")
    return fig


def render_memory_breakdown(result) -> go.Figure:
    """Render pie chart of memory breakdown."""
    labels = ["模型权重", "KV Cache", "其他"]
    values = [result.weight_memory_gb, result.kv_cache_gb,
              max(0, result.memory_gb - result.weight_memory_gb - result.kv_cache_gb)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="显存占用分布")
    return fig


def render_sensitivity_curve(x_values, y_values, x_label, y_label, title) -> go.Figure:
    """Render line chart for sensitivity analysis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=y_label))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig
```

- [ ] **Step 2: 创建 Streamlit 主入口**

`python/web/app.py`:
```python
import streamlit as st

st.set_page_config(
    page_title="算力等效建模评估工具",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 模型对算力等效建模评估工具")
st.markdown("基于异构算力资源池，支持多负载的算力需求表征和硬件推荐。")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能",
    ["算力估算", "多硬件对比", "敏感性分析", "管理"],
    index=0,
)

if page == "算力估算":
    from python.web.pages.estimation import render
    render()
elif page == "多硬件对比":
    from python.web.pages.comparison import render
    render()
elif page == "敏感性分析":
    from python.web.pages.sensitivity import render
    render()
elif page == "管理":
    from python.web.pages.management import render
    render()
```

- [ ] **Step 3: 实现算力估算页面**

`python/web/pages/estimation.py`:
```python
import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager
from python.web.components.charts import render_throughput_bar, render_memory_breakdown

import sys
from pathlib import Path

_build_path = Path(__file__).parent.parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    mc = None


def render():
    if mc is None:
        st.error("C++ 模块未编译。请先运行: `bash scripts/build.sh`")
        return

    analyzer = ModelAnalyzer()
    hw_db = HardwareDB()
    cal_mgr = CalibrationManager()

    st.header("算力估算")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("输入参数")

        model_type = st.selectbox(
            "负载类型",
            ["dense", "moe", "o1_reasoning", "multimodal"],
            format_func=lambda x: {
                "dense": "稠密模型 (Dense)",
                "moe": "MoE 模型",
                "o1_reasoning": "类o1推理模型",
                "multimodal": "多模态模型",
            }[x],
        )

        # Preset selection
        presets = analyzer.list_presets()
        preset_names = presets.get(model_type, [])
        preset_name = st.selectbox("模型预设", ["自定义"] + preset_names)

        if preset_name == "自定义":
            param_billions = st.number_input("参数量 (B)", min_value=0.1, value=7.0, step=0.1)
        else:
            param_billions = None

        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"])
        concurrency = st.slider("并发量", min_value=1, max_value=500, value=1)
        max_tokens = st.slider("最大 tokens 数", min_value=512, max_value=32768, value=2048, step=512)

        # Conditional params
        reasoning_depth = 0
        image_resolution = 0
        num_images = 1

        if model_type == "o1_reasoning":
            reasoning_depth = st.selectbox("推理深度", [1, 2, 3], format_func=lambda x: {1: "轻度 (2-3x)", 2: "中度 (3-5x)", 3: "重度 (5-10x)"}[x])

        if model_type == "multimodal":
            image_resolution = st.selectbox("图像分辨率", [224, 336, 448], index=1)
            num_images = st.slider("图像数量", 1, 10, 1)

        run_button = st.button("开始估算", type="primary", use_container_width=True)

    if run_button:
        with col_result:
            st.subheader("估算结果")

            # Create model params
            params = analyzer.create_params(
                model_type=model_type,
                preset_name=preset_name if preset_name != "自定义" else None,
                param_billions=param_billions,
                quant=quant,
                concurrency=concurrency,
                max_tokens=max_tokens,
                reasoning_depth=reasoning_depth,
                image_resolution=image_resolution,
                num_images=num_images,
            )

            # Run estimation
            engine = mc.EstimationEngine()
            result = engine.estimate(params)

            # Apply calibration
            # (will use default 1.0 factors if no calibration data exists)

            # Display estimation results
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总显存需求", f"{result.memory_gb:.1f} GB")
            m2.metric("权重显存", f"{result.weight_memory_gb:.1f} GB")
            m3.metric("KV Cache", f"{result.kv_cache_gb:.1f} GB")
            m4.metric("总 FLOPs", f"{result.flops_total:.2e}")

            # Memory breakdown chart
            fig_mem = render_memory_breakdown(result)
            st.plotly_chart(fig_mem, use_container_width=True)

            # Hardware matching
            st.subheader("硬件推荐")
            matcher = mc.HardwareMatcher()
            hw_pool = hw_db.to_cpp_hardware_list()
            configs = matcher.match(result, params, hw_pool, baseline_throughput=10.0)

            if configs:
                rows = []
                for c in configs:
                    # Apply calibration
                    adj_tp = cal_mgr.adjust_throughput(c.estimated_throughput, model_type, c.hardware.name)
                    adj_mem = cal_mgr.adjust_memory(result.memory_gb, model_type, c.hardware.name)

                    rows.append({
                        "硬件型号": c.hardware.name,
                        "卡数": c.num_cards,
                        "单卡显存": f"{c.hardware.memory_gb} GB",
                        "预估吞吐": f"{adj_tp:.1f} tokens/s",
                        "预估延迟": f"{c.estimated_latency_ms:.1f} ms",
                        "瓶颈类型": c.bottleneck_type,
                        "并行策略": c.parallel_strategy,
                        "满足基线": "✅" if c.meets_baseline else "❌",
                    })

                st.table(rows)

                # Throughput comparison chart
                chart_data = [{"hardware": r["硬件型号"], "throughput": float(r["预估吞吐"].split()[0]), "meets_baseline": "✅" in r["满足基线"]} for r in rows]
                fig_tp = render_throughput_bar(chart_data)
                st.plotly_chart(fig_tp, use_container_width=True)
            else:
                st.warning("未找到匹配的硬件配置。")
```

- [ ] **Step 4: 创建占位页面**

`python/web/pages/comparison.py`:
```python
import streamlit as st


def render():
    st.header("多硬件对比")
    st.info("此功能将在后续任务中实现。")
```

`python/web/pages/sensitivity.py`:
```python
import streamlit as st


def render():
    st.header("敏感性分析")
    st.info("此功能将在后续任务中实现。")
```

`python/web/pages/management.py`:
```python
import streamlit as st


def render():
    st.header("管理")
    st.info("此功能将在后续任务中实现。")
```

- [ ] **Step 5: 验证主页面可以运行**

```bash
streamlit run python/web/app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 | head -20
```

Expected: Streamlit 服务启动成功，返回 HTML 内容。

- [ ] **Step 6: 提交**

```bash
git add python/web/
git commit -m "feat: implement Streamlit estimation page with hardware matching"
```

---

## Task 10: Streamlit Web UI — 多硬件对比与敏感性分析

**Files:**
- Modify: `python/web/pages/comparison.py`
- Modify: `python/web/pages/sensitivity.py`

- [ ] **Step 1: 实现多硬件对比页面**

`python/web/pages/comparison.py`:
```python
import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.web.components.charts import render_throughput_bar

import sys
from pathlib import Path

_build_path = Path(__file__).parent.parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    mc = None


def render():
    if mc is None:
        st.error("C++ 模块未编译。请先运行: `bash scripts/build.sh`")
        return

    analyzer = ModelAnalyzer()
    hw_db = HardwareDB()

    st.header("多硬件对比")
    st.markdown("输入模型参数，同时对比所有可用硬件的性能表现。")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("负载类型", ["dense", "moe", "o1_reasoning", "multimodal"],
                                  format_func=lambda x: {"dense": "稠密", "moe": "MoE", "o1_reasoning": "类o1", "multimodal": "多模态"}[x], key="cmp_type")
        presets = analyzer.list_presets()
        preset_name = st.selectbox("模型预设", ["自定义"] + presets.get(model_type, []), key="cmp_preset")
        if preset_name == "自定义":
            param_billions = st.number_input("参数量 (B)", 0.1, value=7.0, key="cmp_params")
        else:
            param_billions = None
    with col2:
        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"], key="cmp_quant")
        concurrency = st.slider("并发量", 1, 500, 1, key="cmp_conc")
        max_tokens = st.slider("最大 tokens", 512, 32768, 2048, step=512, key="cmp_tokens")

    if st.button("开始对比", type="primary", use_container_width=True):
        params = analyzer.create_params(
            model_type=model_type,
            preset_name=preset_name if preset_name != "自定义" else None,
            param_billions=param_billions,
            quant=quant, concurrency=concurrency, max_tokens=max_tokens,
        )

        engine = mc.EstimationEngine()
        result = engine.estimate(params)
        matcher = mc.HardwareMatcher()
        hw_pool = hw_db.to_cpp_hardware_list()
        configs = matcher.match(result, params, hw_pool, 10.0)

        if not configs:
            st.warning("未找到匹配的硬件。")
            return

        # Results table
        rows = []
        for c in configs:
            rows.append({
                "硬件": c.hardware.name,
                "厂商": c.hardware.vendor,
                "卡数": c.num_cards,
                "单卡显存 (GB)": c.hardware.memory_gb,
                "预估吞吐 (tokens/s)": round(c.estimated_throughput, 1),
                "预估延迟 (ms)": round(c.estimated_latency_ms, 1),
                "瓶颈": c.bottleneck_type,
                "并行策略": c.parallel_strategy,
                "满足基线": c.meets_baseline,
            })

        st.dataframe(rows, use_container_width=True)

        # Chart
        chart_data = [{"hardware": r["硬件"], "throughput": r["预估吞吐 (tokens/s)"], "meets_baseline": r["满足基线"]} for r in rows]
        fig = render_throughput_bar(chart_data)
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: 实现敏感性分析页面**

`python/web/pages/sensitivity.py`:
```python
import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.web.components.charts import render_sensitivity_curve

import sys
from pathlib import Path

_build_path = Path(__file__).parent.parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    mc = None


def render():
    if mc is None:
        st.error("C++ 模块未编译。请先运行: `bash scripts/build.sh`")
        return

    analyzer = ModelAnalyzer()
    hw_db = HardwareDB()

    st.header("敏感性分析")
    st.markdown("分析不同参数变化对算力需求和预估性能的影响。")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("负载类型", ["dense", "moe", "o1_reasoning", "multimodal"],
                                  format_func=lambda x: {"dense": "稠密", "moe": "MoE", "o1_reasoning": "类o1", "multimodal": "多模态"}[x], key="sa_type")
        presets = analyzer.list_presets()
        preset_name = st.selectbox("模型预设", ["自定义"] + presets.get(model_type, []), key="sa_preset")
        if preset_name == "自定义":
            param_billions = st.number_input("参数量 (B)", 0.1, value=7.0, key="sa_params")
        else:
            param_billions = None
        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"], key="sa_quant")
    with col2:
        hardware_name = st.selectbox("目标硬件", [h["name"] for h in hw_db.list_hardware()], key="sa_hw")
        max_tokens = st.slider("最大 tokens", 512, 32768, 2048, step=512, key="sa_tokens")

    analysis_var = st.selectbox("分析变量", ["并发量", "参数量", "序列长度"], key="sa_var")

    if st.button("开始分析", type="primary", use_container_width=True):
        hw = hw_db.get_hardware(hardware_name)
        if not hw:
            st.error("硬件未找到")
            return

        hw_spec = hw_db.to_cpp_hardware_list()
        target_hw = [h for h in hw_spec if h.name == hardware_name]
        if not target_hw:
            st.error("硬件数据错误")
            return

        engine = mc.EstimationEngine()
        matcher = mc.HardwareMatcher()

        x_values = []
        y_throughput = []
        y_memory = []
        y_cards = []

        if analysis_var == "并发量":
            x_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
            for conc in x_range:
                params = analyzer.create_params(
                    model_type=model_type,
                    preset_name=preset_name if preset_name != "自定义" else None,
                    param_billions=param_billions,
                    quant=quant, concurrency=conc, max_tokens=max_tokens,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    x_values.append(conc)
                    y_throughput.append(configs[0].estimated_throughput)
                    y_memory.append(result.memory_gb)
                    y_cards.append(configs[0].num_cards)
            x_label = "并发量"

        elif analysis_var == "参数量":
            x_range = [1.0, 3.0, 7.0, 13.0, 30.0, 70.0, 130.0]
            for pb in x_range:
                params = analyzer.create_params(
                    model_type="dense", param_billions=pb,
                    quant=quant, concurrency=1, max_tokens=max_tokens,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    x_values.append(pb)
                    y_throughput.append(configs[0].estimated_throughput)
                    y_memory.append(result.memory_gb)
                    y_cards.append(configs[0].num_cards)
            x_label = "参数量 (B)"

        else:  # 序列长度
            x_range = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            for seq in x_range:
                params = analyzer.create_params(
                    model_type=model_type,
                    preset_name=preset_name if preset_name != "自定义" else None,
                    param_billions=param_billions,
                    quant=quant, concurrency=1, max_tokens=seq,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    x_values.append(seq)
                    y_throughput.append(configs[0].estimated_throughput)
                    y_memory.append(result.memory_gb)
                    y_cards.append(configs[0].num_cards)
            x_label = "最大序列长度"

        if x_values:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig1 = render_sensitivity_curve(x_values, y_throughput, x_label, "吞吐量 (tokens/s)", "吞吐量变化")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = render_sensitivity_curve(x_values, y_memory, x_label, "显存 (GB)", "显存需求变化")
                st.plotly_chart(fig2, use_container_width=True)
            with c3:
                fig3 = render_sensitivity_curve(x_values, y_cards, x_label, "卡数", "卡数变化")
                st.plotly_chart(fig3, use_container_width=True)
```

- [ ] **Step 3: 验证页面运行**

```bash
streamlit run python/web/app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 >/dev/null && echo "OK"
```

Expected: 返回 OK。

- [ ] **Step 4: 提交**

```bash
git add python/web/pages/comparison.py python/web/pages/sensitivity.py
git commit -m "feat: implement comparison and sensitivity analysis pages"
```

---

## Task 11: Streamlit Web UI — 管理页面

**Files:**
- Modify: `python/web/pages/management.py`

- [ ] **Step 1: 实现管理页面**

`python/web/pages/management.py`:
```python
import streamlit as st
import pandas as pd
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager


def render():
    hw_db = HardwareDB()
    cal_mgr = CalibrationManager()
    cal_mgr.load()

    st.header("系统管理")

    tab_hw, tab_cal, tab_about = st.tabs(["硬件数据库", "校准数据", "关于"])

    with tab_hw:
        st.subheader("硬件数据库管理")

        # List existing hardware
        hw_list = hw_db.list_hardware()
        if hw_list:
            df = pd.DataFrame([
                {
                    "名称": h["name"],
                    "厂商": h["vendor"],
                    "类型": h["type"],
                    "FP16 TFLOPS": h["specs"]["fp16_tflops"],
                    "显存 (GB)": h["specs"]["memory_gb"],
                    "带宽 (GB/s)": h["specs"]["memory_bandwidth_gbs"],
                }
                for h in hw_list
            ])
            st.dataframe(df, use_container_width=True)

        # Add new hardware
        st.markdown("---")
        st.subheader("添加新硬件")
        with st.form("add_hardware"):
            name = st.text_input("硬件名称")
            vendor = st.text_input("厂商")
            hw_type = st.selectbox("类型", ["GPU", "NPU"])
            col1, col2, col3 = st.columns(3)
            with col1:
                fp16 = st.number_input("FP16 TFLOPS", 0.0, value=100.0)
                mem = st.number_input("显存 (GB)", 0.0, value=32.0)
            with col2:
                int8 = st.number_input("INT8 TOPS", 0.0, value=200.0)
                bw = st.number_input("带宽 (GB/s)", 0.0, value=1000.0)
            with col3:
                nvlink = st.number_input("NVLink 带宽 (GB/s)", 0.0, value=0.0)
                cost = st.number_input("参考价格", 0.0, value=5000.0)

            if st.form_submit_button("添加"):
                hw = {
                    "name": name, "vendor": vendor, "architecture": "", "type": hw_type,
                    "specs": {
                        "fp16_tflops": fp16, "int8_tops": int8, "fp32_tflops": 0,
                        "memory_gb": mem, "memory_type": "", "memory_bandwidth_gbs": bw,
                        "nvlink_bandwidth_gbs": nvlink, "pcie_version": "", "max_tdp_watts": 0,
                    },
                    "cost_per_unit": cost, "notes": "",
                }
                hw_db.add_hardware(hw)
                hw_db.save()
                st.success(f"已添加: {name}")
                st.rerun()

        # Remove hardware
        st.markdown("---")
        st.subheader("删除硬件")
        del_name = st.selectbox("选择要删除的硬件", [h["name"] for h in hw_list])
        if st.button("删除", type="secondary"):
            hw_db.remove_hardware(del_name)
            hw_db.save()
            st.success(f"已删除: {del_name}")
            st.rerun()

    with tab_cal:
        st.subheader("校准数据管理")

        entries = cal_mgr.list_entries()
        if entries:
            df = pd.DataFrame(entries)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无校准数据。可以通过 CSV 文件导入或在下方手动添加。")

        # Import CSV
        st.markdown("---")
        st.subheader("导入校准数据 (CSV)")
        uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
        if uploaded:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(uploaded.getvalue().decode())
                tmp_path = f.name
            count = cal_mgr.import_csv(tmp_path)
            os.unlink(tmp_path)
            cal_mgr.save()
            st.success(f"成功导入 {count} 条校准记录")
            st.rerun()

        # Manual add
        st.markdown("---")
        st.subheader("手动添加校准记录")
        with st.form("add_calibration"):
            col1, col2 = st.columns(2)
            with col1:
                mt = st.selectbox("模型类型", ["dense", "moe", "o1_reasoning", "multimodal"], key="cal_mt")
                hw_name = st.text_input("硬件名称", key="cal_hw")
            with col2:
                pred_tp = st.number_input("预测吞吐", 0.0, value=20.0, key="cal_pred_tp")
                act_tp = st.number_input("实际吞吐", 0.0, value=16.0, key="cal_act_tp")
                pred_mem = st.number_input("预测显存", 0.0, value=14.0, key="cal_pred_mem")
                act_mem = st.number_input("实际显存", 0.0, value=15.0, key="cal_act_mem")
            if st.form_submit_button("添加"):
                cal_mgr.add_point(mt, hw_name, pred_tp, act_tp, pred_mem, act_mem)
                cal_mgr.save()
                st.success("校准记录已添加")
                st.rerun()

    with tab_about:
        st.subheader("关于本工具")
        st.markdown("""
        **模型对算力等效建模评估工具 v1.0**

        支持的模型类型：
        - 稠密模型 (Dense): LLaMA, GPT, Qwen 等
        - MoE 模型: Mixtral, DeepSeek-V2 等
        - 类o1推理模型: DeepSeek-R1 等
        - 多模态模型: LLaVA, Qwen-VL 等

        支持的硬件：
        - NVIDIA GPU: A100, H100, H200, L40S, RTX 4090
        - 昇腾 NPU: 910B, 300I Duo
        - 其他国产: 寒武纪 MLU370

        **架构:** C++ 核心引擎 + Python 业务层 + Streamlit Web UI
        """)
```

- [ ] **Step 2: 验证管理页面**

```bash
streamlit run python/web/app.py --server.headless true &
sleep 3
curl -s http://localhost:8501 >/dev/null && echo "OK"
```

Expected: 返回 OK。

- [ ] **Step 3: 提交**

```bash
git add python/web/pages/management.py
git commit -m "feat: implement management page with hardware and calibration CRUD"
```

---

## Task 12: 文档编写

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/formulas.md`
- Modify: `docs/usage.md`
- Modify: `docs/calibration_guide.md`

- [ ] **Step 1: 编写架构文档**

`docs/architecture.md`:
```markdown
# 系统架构

## 三层架构

本工具采用 C++ + Python 混合架构：

1. **计算引擎层 (C++)**: FLOPs/显存/带宽计算、硬件匹配、校准拟合
2. **业务逻辑层 (Python)**: 模型分析、数据管理、参数验证
3. **展示层 (Streamlit)**: Web UI、图表、报告导出

通过 pybind11 桥接 C++ 和 Python。

## 目录结构

- `cpp/` — C++ 核心引擎源码
- `python/` — Python 业务层和 Web UI
- `docs/` — 项目文档
- `scripts/` — 构建和测试脚本
```

- [ ] **Step 2: 编写公式文档**

`docs/formulas.md`:
```markdown
# 估算公式

## Dense 模型

### 显存
- 模型权重 = 参数量 × 每参数字节数 (FP16=2, INT8=1, INT4=0.5)
- KV Cache = 2 × 层数 × hidden_dim × seq_len × 并发量 × 每参数字节数
- 总显存 = (权重 + KV Cache + 激活值) × 1.10

### FLOPs
- Prefill = 2 × 参数量 × 输入序列长度
- Decode (每token) = 2 × 参数量

### 吞吐量
- 计算受限 = TFLOPS × 10^12 / (2 × 参数量)
- 带宽受限 = 带宽 × 10^9 / (参数量 × 每参数字节数)

## MoE 模型
- 显存按总参数量计算
- FLOPs 按激活参数量计算

## 类o1推理模型
- 在 Dense 基础上乘以推理 token 倍率 (2x~10x)

## 多模态模型
- 语言模型 + 视觉编码器分开估算

## 参考文献
- Megatron-LM (arXiv:1909.08053)
- Switch Transformers (Fedus et al., 2022)
- DeepSeek-V2 (arXiv:2405.04434)
```

- [ ] **Step 3: 编写使用文档**

`docs/usage.md`:
```markdown
# 使用说明

## 安装

```bash
# 1. 编译 C++ 模块
bash scripts/build.sh

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装项目
pip install -e python/
```

## 启动 Web UI

```bash
streamlit run python/web/app.py
```

浏览器打开 http://localhost:8501

## 功能说明

### 算力估算
输入模型参数，获取推荐硬件配置和性能预估。

### 多硬件对比
同时对比所有可用硬件的性能表现。

### 敏感性分析
分析并发量/参数量/序列长度变化对性能的影响。

### 管理
管理硬件数据库和校准数据。

## Python API

```python
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB

import model_compute as mc

analyzer = ModelAnalyzer()
params = analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)

engine = mc.EstimationEngine()
result = engine.estimate(params)
print(f"显存需求: {result.memory_gb:.1f} GB")
```
```

- [ ] **Step 4: 编写校准指南**

`docs/calibration_guide.md`:
```markdown
# 校准指南

## 为什么需要校准？

理论公式可能与实际性能存在系统性偏差。校准模块通过少量实测数据修正这些偏差。

## 校准流程

1. 在目标硬件上运行模型推理
2. 记录实际吞吐量和显存占用
3. 与工具的预测值对比
4. 导入校准数据
5. 工具自动应用校准系数

## 校准数据格式 (CSV)

```csv
# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem
dense,NVIDIA A100 80GB,20.0,16.0,14.0,15.5
moe,NVIDIA H100 80GB,50.0,42.0,60.0,65.0
```

## 导入校准数据

1. Web UI → 管理 → 校准数据 → 上传 CSV
2. 或使用 Python API:
   ```python
   from python.core.calibration_mgr import CalibrationManager
   mgr = CalibrationManager()
   mgr.import_csv("path/to/calibration.csv")
   mgr.save()
   ```

## 注意事项

- 校准系数按 (模型类型, 硬件类型) 分组
- 未校准的配置使用默认系数 1.0
- 建议每个组合至少 3 个数据点
```

- [ ] **Step 5: 提交**

```bash
git add docs/
git commit -m "docs: add architecture, formulas, usage, and calibration guide"
```

---

## Task 13: 全面测试与校准验证

**Files:**
- Modify: `python/tests/test_integration.py`

- [ ] **Step 1: 编写集成测试**

`python/tests/test_integration.py`:
```python
"""Integration tests: end-to-end estimation pipeline."""
import pytest
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager

import sys
from pathlib import Path
_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    mc = None


@pytest.mark.skipif(mc is None, reason="C++ module not built")
class TestIntegration:
    def setup_method(self):
        self.analyzer = ModelAnalyzer()
        self.hw_db = HardwareDB()
        self.cal_mgr = CalibrationManager()
        self.engine = mc.EstimationEngine()
        self.matcher = mc.HardwareMatcher()

    def test_dense_7b_estimation(self):
        params = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        result = self.engine.estimate(params)
        assert result.memory_gb > 10
        assert result.memory_gb < 30
        assert result.weight_memory_gb > 10
        assert result.flops_total > 0

    def test_dense_70b_needs_multi_card(self):
        params = self.analyzer.create_params("dense", "LLaMA-2 70B", "FP16", 1, 4096)
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        a100_configs = [c for c in configs if "A100" in c.hardware.name and "80" in c.hardware.name]
        if a100_configs:
            assert a100_configs[0].num_cards >= 2

    def test_int8_reduces_memory(self):
        params_fp16 = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        params_int8 = self.analyzer.create_params("dense", "LLaMA-2 7B", "INT8", 1, 2048)
        r_fp16 = self.engine.estimate(params_fp16)
        r_int8 = self.engine.estimate(params_int8)
        assert r_int8.weight_memory_gb < r_fp16.weight_memory_gb

    def test_moe_sparse_advantage(self):
        params = self.analyzer.create_params("moe", "Mixtral 8x7B", "FP16", 1, 2048)
        result = self.engine.estimate(params)
        # MoE 46.7B total but only ~12B active
        assert result.memory_gb > 40  # All experts stored
        assert result.flops_total > 0

    def test_o1_reasoning_increases_memory(self):
        params_normal = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        params_o1 = self.analyzer.create_params("o1_reasoning", "DeepSeek-R1", "FP16", 1, 4096, reasoning_depth=3)
        r_normal = self.engine.estimate(params_normal)
        r_o1 = self.engine.estimate(params_o1)
        # o1 with heavy reasoning should need more memory
        assert r_o1.memory_gb > r_normal.memory_gb

    def test_multimodal_adds_vision_memory(self):
        params_text = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        params_mm = self.analyzer.create_params("multimodal", "LLaVA-1.5 7B", "FP16", 1, 2048, image_resolution=336, num_images=1)
        r_text = self.engine.estimate(params_text)
        r_mm = self.engine.estimate(params_mm)
        assert r_mm.memory_gb > r_text.memory_gb

    def test_high_concurrency_increases_memory(self):
        params_low = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        params_high = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 64, 2048)
        r_low = self.engine.estimate(params_low)
        r_high = self.engine.estimate(params_high)
        assert r_high.memory_gb > r_low.memory_gb

    def test_hardware_matching_returns_sorted(self):
        params = self.analyzer.create_params("dense", "LLaMA-2 7B", "FP16", 1, 2048)
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        assert len(configs) >= 2
        # Should be sorted by throughput descending
        for i in range(len(configs) - 1):
            assert configs[i].estimated_throughput >= configs[i + 1].estimated_throughput

    def test_calibration_adjusts_prediction(self):
        self.cal_mgr.add_point("dense", "A100", 20.0, 16.0, 14.0, 15.0)
        adjusted = self.cal_mgr.adjust_throughput(30.0, "dense", "A100")
        assert abs(adjusted - 24.0) < 0.1  # 30 * 0.8

    def test_full_pipeline(self):
        """Full pipeline: create params → estimate → match → calibrate."""
        params = self.analyzer.create_params("dense", "LLaMA-2 13B", "INT8", 8, 4096)
        result = self.engine.estimate(params)

        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        assert len(configs) > 0

        # Apply calibration (default factor 1.0)
        best = configs[0]
        adj_tp = self.cal_mgr.adjust_throughput(best.estimated_throughput, "dense", best.hardware.name)
        assert adj_tp > 0
```

- [ ] **Step 2: 运行全部测试**

```bash
bash scripts/run_tests.sh
```

Expected: 所有 C++ 和 Python 测试全部通过。

- [ ] **Step 3: 提交**

```bash
git add python/tests/test_integration.py
git commit -m "test: add integration tests for full estimation pipeline"
```

---

## Task 14: 最终检查与清理

- [ ] **Step 1: 运行完整测试套件，确保全部通过**

```bash
bash scripts/run_tests.sh
```

Expected: 所有测试 PASS。

- [ ] **Step 2: 启动 Web UI，手动验证所有页面功能**

```bash
streamlit run python/web/app.py
```

手动验证：
- [ ] 算力估算页面：选择 Dense 7B → 获得硬件推荐
- [ ] 算力估算页面：选择 MoE → 获得硬件推荐
- [ ] 算力估算页面：选择 o1 → 获得硬件推荐
- [ ] 算力估算页面：选择 多模态 → 获得硬件推荐
- [ ] 多硬件对比页面：查看所有硬件对比
- [ ] 敏感性分析页面：调整并发量查看曲线
- [ ] 管理页面：添加/删除硬件
- [ ] 管理页面：导入校准数据

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```

---

## 完成

所有任务完成后，项目应具备：

1. **C++ 核心引擎**：Dense/MoE/o1/多模态四种模型的 FLOPs/显存/带宽估算
2. **硬件匹配器**：自动匹配最优硬件配置，支持并行策略选择
3. **校准模块**：用实测数据修正理论预测
4. **Python 业务层**：模型分析器、硬件数据库、校准管理器
5. **Streamlit Web UI**：4 个功能页面（估算、对比、敏感性分析、管理）
6. **完整文档**：架构、公式、使用说明、校准指南
7. **测试覆盖**：C++ 单元测试 + Python 单元测试 + 集成测试
