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
        params = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        assert result.memory_gb > 10
        assert result.memory_gb < 30
        assert result.weight_memory_gb > 10
        assert result.flops_total > 0

    def test_dense_70b_needs_multi_card(self):
        params = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 70B",
            quant="FP16",
            concurrency=1,
            max_tokens=4096,
        )
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        a100_configs = [c for c in configs if "A100" in c.hardware.name and "80" in c.hardware.name]
        if a100_configs:
            assert a100_configs[0].num_cards >= 2

    def test_int8_reduces_memory(self):
        params_fp16 = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_int8 = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="INT8",
            concurrency=1,
            max_tokens=2048,
        )
        r_fp16 = self.engine.estimate(params_fp16)
        r_int8 = self.engine.estimate(params_int8)
        assert r_int8.weight_memory_gb < r_fp16.weight_memory_gb

    def test_moe_sparse_advantage(self):
        params = self.analyzer.create_params(
            model_type="moe",
            preset_name="Mixtral 8x7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        assert result.memory_gb > 40  # All experts stored
        assert result.flops_total > 0

    def test_o1_reasoning_increases_memory(self):
        params_normal = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_o1 = self.analyzer.create_params(
            model_type="o1_reasoning",
            preset_name="DeepSeek-R1",
            quant="FP16",
            concurrency=1,
            max_tokens=4096,
            reasoning_depth=3,
        )
        r_normal = self.engine.estimate(params_normal)
        r_o1 = self.engine.estimate(params_o1)
        assert r_o1.memory_gb > r_normal.memory_gb

    def test_multimodal_adds_vision_memory(self):
        params_text = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_mm = self.analyzer.create_params(
            model_type="multimodal",
            preset_name="LLaVA-1.5 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
            image_resolution=336,
            num_images=1,
        )
        r_text = self.engine.estimate(params_text)
        r_mm = self.engine.estimate(params_mm)
        assert r_mm.memory_gb > r_text.memory_gb

    def test_high_concurrency_increases_memory(self):
        params_low = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_high = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=64,
            max_tokens=2048,
        )
        r_low = self.engine.estimate(params_low)
        r_high = self.engine.estimate(params_high)
        assert r_high.memory_gb > r_low.memory_gb

    def test_hardware_matching_returns_sorted(self):
        params = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 7B",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        assert len(configs) >= 2
        for i in range(len(configs) - 1):
            assert configs[i].estimated_throughput >= configs[i + 1].estimated_throughput

    def test_calibration_adjusts_prediction(self):
        self.cal_mgr.add_point("dense", "A100", 20.0, 16.0, 14.0, 15.0)
        factor = self.cal_mgr.get_factor("dense", "A100")
        adjusted = 30.0 * factor.throughput_factor
        assert abs(adjusted - 24.0) < 0.1  # 30 * 0.8

    def test_full_pipeline(self):
        """Full pipeline: create params -> estimate -> match -> calibrate."""
        params = self.analyzer.create_params(
            model_type="dense",
            preset_name="LLaMA-2 13B",
            quant="INT8",
            concurrency=8,
            max_tokens=4096,
        )
        result = self.engine.estimate(params)

        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        assert len(configs) > 0

        best = configs[0]
        factor = self.cal_mgr.get_factor("dense", best.hardware.name)
        adj_tp = best.estimated_throughput * factor.throughput_factor
        assert adj_tp > 0

    def test_recommendation_dlrm_estimation(self):
        params = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-small",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        assert result.memory_gb > 0.5
        assert result.flops_total > 0

    def test_recommendation_dlrm_large_more_memory(self):
        params_small = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-small",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_large = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-large",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        r_small = self.engine.estimate(params_small)
        r_large = self.engine.estimate(params_large)
        assert r_large.memory_gb > r_small.memory_gb

    def test_recommendation_int8_less_memory(self):
        params_fp16 = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-small",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        params_int8 = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-small",
            quant="INT8",
            concurrency=1,
            max_tokens=2048,
        )
        r_fp16 = self.engine.estimate(params_fp16)
        r_int8 = self.engine.estimate(params_int8)
        assert r_int8.weight_memory_gb < r_fp16.weight_memory_gb

    def test_recommendation_hardware_matching(self):
        params = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="DLRM-small",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        configs = self.matcher.match(result, params, hw_pool, 10.0)
        assert len(configs) > 0

    def test_recommendation_sasrec(self):
        params = self.analyzer.create_params(
            model_type="recommendation",
            preset_name="SASRec",
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = self.engine.estimate(params)
        assert result.memory_gb > 0
