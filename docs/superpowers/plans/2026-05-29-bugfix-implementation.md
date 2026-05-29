# Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 15 bugs found during code review: 5 critical, 5 high, 5 medium.

**Architecture:** Fix C++ formula bugs (overflow, KV cache precision, activation memory, compute scaling), fix Python data corruption and UI bugs, add missing features.

**Tech Stack:** C++17, Python 3.10+, Streamlit

---

## Task 1: C++ Critical Fixes (Findings 1-4)

**Files:**
- Modify: `cpp/src/estimation_engine.cpp`
- Modify: `cpp/src/hardware_matcher.cpp`
- Modify: `cpp/tests/test_engine.cpp`

- [ ] **Step 1: Fix integer overflow in activation memory (Finding 1)**

In `estimate_dense()`, line 72, change:
```cpp
    double activation_bytes = p.concurrency * p.max_tokens * hidden_dim * num_layers * bpp * activation_factor;
```
to:
```cpp
    double activation_bytes = static_cast<double>(p.concurrency) * p.max_tokens * hidden_dim * num_layers * bpp * activation_factor;
```

- [ ] **Step 2: Fix KV cache using weight quantization (Finding 2)**

In `estimate_dense()`, KV cache should always use FP16 (2 bytes/element), not weight bpp. Change the kv_bytes line to use a fixed 2.0 for the bytes-per-element:
```cpp
    double kv_bpe = 2.0;  // KV cache always FP16 during inference
    double kv_bytes = 2.0 * num_layers * kv_dim * p.max_tokens * p.concurrency * kv_bpe;
```

Also apply the same fix in `estimate_moe()`.

- [ ] **Step 3: Fix activation memory multiplied by num_layers (Finding 4)**

Activation memory during inference is per-layer (not all layers simultaneously). Change:
```cpp
    double activation_bytes = static_cast<double>(p.concurrency) * p.max_tokens * hidden_dim * num_layers * bpp * activation_factor;
```
to:
```cpp
    double activation_bytes = static_cast<double>(p.concurrency) * p.max_tokens * hidden_dim * bpp * activation_factor;
```
(Remove `* num_layers`)

- [ ] **Step 4: Fix compute throughput missing N factor (Finding 3)**

In `hardware_matcher.cpp`, change the compute throughput line to account for tensor parallelism:
```cpp
        double compute_throughput = (model_params.param_billions > 0)
            ? compute_tflops * 1e12 * config.num_cards / (2.0 * model_params.param_billions * 1e9)
            : 1e18;
```

- [ ] **Step 5: Build and run tests, fix any failures**

```bash
cd build && cmake --build . --target test_engine && ./cpp/test_engine.exe
```

Adjust test tolerances as needed as the formulas changed significantly.

- [ ] **Step 6: Commit**

```bash
git add cpp/src/estimation_engine.cpp cpp/src/hardware_matcher.cpp cpp/tests/test_engine.cpp
git commit -m "fix: critical formula bugs - int overflow, KV cache precision, activation memory, compute scaling"
```

---

## Task 2: C++ High Fixes (Findings 7-10)

**Files:**
- Modify: `cpp/src/estimation_engine.cpp`
- Modify: `cpp/tests/test_engine.cpp`

- [ ] **Step 1: Fix MoE missing activation memory (Finding 8)**

In `estimate_moe()`, add activation memory calculation before `total_bytes`:
```cpp
    double activation_bytes = static_cast<double>(p.concurrency) * p.max_tokens * hidden_dim * bpp * 2.0;
    double total_bytes = total_params_bytes + kv_bytes + activation_bytes;
```

- [ ] **Step 2: Fix multimodal image KV cache (Finding 9)**

In `estimate_multimodal()`, change the image KV cache to use kv_dim:
```cpp
    int effective_kv_heads = (p.num_kv_heads > 0) ? p.num_kv_heads : num_heads;
    int effective_head_dim = (p.head_dim > 0) ? p.head_dim : (hidden_dim / num_heads);
    int kv_dim = effective_kv_heads * effective_head_dim;
    double img_kv_bytes = 2.0 * num_layers * kv_dim * img_tokens * p.num_images * p.concurrency * 2.0;
```

- [ ] **Step 3: Improve MoE active parameter estimation (Finding 10)**

In `estimate_moe()`, account for shared parameters:
```cpp
    // Shared params (attention, embedding, norms) are always ~30% of total
    double shared_ratio = 0.30;
    double shared_params_b = p.param_billions * shared_ratio;
    double expert_params_b = p.param_billions * (1.0 - shared_ratio);
    double active_expert_params_b = expert_params_b * active_ratio;
    double active_params_b = shared_params_b + active_expert_params_b;
```

- [ ] **Step 4: Build, test, commit**

```bash
cd build && cmake --build . --target test_engine && ./cpp/test_engine.exe
git add cpp/src/estimation_engine.cpp cpp/tests/test_engine.cpp
git commit -m "fix: MoE activation memory, multimodal GQA KV cache, active param estimation"
```

---

## Task 3: C++ Medium Fixes (Findings 13-15)

**Files:**
- Modify: `cpp/include/hardware_matcher.h`
- Modify: `cpp/src/estimation_engine.cpp`

- [ ] **Step 1: Add default initializers to HardwareSpec (Finding 14)**

Edit `cpp/include/hardware_matcher.h`, add default values:
```cpp
struct HardwareSpec {
    std::string name;
    std::string vendor;
    std::string architecture;
    std::string type;
    double fp16_tflops = 0;
    double int8_tops = 0;
    double fp32_tflops = 0;
    double memory_gb = 0;
    std::string memory_type;
    double memory_bandwidth_gbs = 0;
    double nvlink_bandwidth_gbs = 0;
    std::string pcie_version;
    double max_tdp_watts = 0;
    double cost_per_unit = 0;
};
```

- [ ] **Step 2: Extract shared architecture inference (Finding 17)**

Move `infer_architecture` to a shared header or make it a non-static function accessible from both files.

- [ ] **Step 3: Commit**

```bash
git add cpp/include/hardware_matcher.h cpp/src/estimation_engine.cpp cpp/src/hardware_matcher.cpp
git commit -m "fix: add default initializers, extract shared architecture inference"
```

---

## Task 4: Python Fixes (Findings 5-6, 11-12)

**Files:**
- Modify: `python/core/calibration_mgr.py`
- Modify: `python/web/pages/estimation.py`
- Modify: `python/web/pages/comparison.py`
- Modify: `python/web/pages/sensitivity.py`
- Modify: `python/web/pages/management.py`

- [ ] **Step 1: Fix calibration save/load data corruption (Finding 5)**

In `calibration_mgr.py`, modify `save()` to only write non-default points:
```python
    def save(self, path: Optional[str] = None):
        save_path = Path(path) if path else self._dir / "calibration.csv"
        # Only save user-added points (not defaults)
        user_points = [p for p in self._points if not p.get("_is_default", False)]
        with open(save_path, "w") as f:
            f.write("# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem\n")
            for p in user_points:
                f.write(f"{p['model_type']},{p['hardware_name']},"
                        f"{p['predicted_throughput']},{p['actual_throughput']},"
                        f"{p['predicted_memory']},{p['actual_memory']}\n")
```

And mark default points in `__init__`:
```python
        if default_csv.exists():
            self.import_csv(str(default_csv))
            for p in self._points:
                p["_is_default"] = True
```

- [ ] **Step 2: Fix preset override bugs (Finding 6)**

In all three web pages, make the conditional params only set values when the model type matches:

For `estimation.py`, change:
```python
        reasoning_depth = 0
        image_resolution = 0
```
to only set them when the model type requires them, and pass `None` otherwise to `create_params()`.

- [ ] **Step 3: Add "recommendation" to calibration management (Finding 11)**

In `management.py`, add "recommendation" to the model_type selectbox.

- [ ] **Step 4: Guard against empty hardware list (Finding 12)**

In `management.py`, wrap the delete selectbox in a check:
```python
        if hw_list:
            del_name = st.selectbox(...)
            if st.button("删除"):
                ...
        else:
            st.info("无硬件可删除。")
```

- [ ] **Step 5: Commit**

```bash
git add python/core/calibration_mgr.py python/web/pages/
git commit -m "fix: calibration data corruption, preset override bugs, missing recommendation type, empty list guard"
```

---

## Task 5: Final Verification

- [ ] **Step 1: Run all tests**
- [ ] **Step 2: Verify no regressions**
- [ ] **Step 3: Final commit if needed**
