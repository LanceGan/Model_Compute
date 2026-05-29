"""Validation tests: compare predictions against known benchmarks."""
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
class TestValidation:
    def setup_method(self):
        self.analyzer = ModelAnalyzer()
        self.hw_db = HardwareDB()
        self.cal_mgr = CalibrationManager()
        self.engine = mc.EstimationEngine()
        self.matcher = mc.HardwareMatcher()

    def _get_throughput(self, model_type, preset_name, quant, hardware_name, concurrency=1, max_tokens=2048):
        params = self.analyzer.create_params(
            model_type=model_type, preset_name=preset_name,
            quant=quant, concurrency=concurrency, max_tokens=max_tokens,
        )
        result = self.engine.estimate(params)
        hw_pool = self.hw_db.to_cpp_hardware_list()
        target_hw = [h for h in hw_pool if h.name == hardware_name]
        if not target_hw:
            return None, result
        configs = self.matcher.match(result, params, target_hw, 10.0)
        if not configs:
            return None, result
        factor = self.cal_mgr.get_factor(model_type, hardware_name)
        adj_tp = configs[0].estimated_throughput * factor.throughput_factor
        return adj_tp, result

    def test_llama2_7b_a100_throughput(self):
        """LLaMA-2 7B FP16 on A100 80G: actual ~15-20 tokens/s"""
        tp, _ = self._get_throughput("dense", "LLaMA-2 7B", "FP16", "NVIDIA A100 80GB")
        assert tp is not None
        assert 5 < tp < 200, f"Throughput {tp:.1f} outside expected range [5, 200]"

    def test_llama2_70b_a100_memory(self):
        """LLaMA-2 70B FP16: actual memory ~140-160 GB"""
        params = self.analyzer.create_params(
            model_type="dense", preset_name="LLaMA-2 70B",
            quant="FP16", concurrency=1, max_tokens=4096,
        )
        result = self.engine.estimate(params)
        factor = self.cal_mgr.get_factor("dense", "NVIDIA A100 80GB")
        adj_mem = result.memory_gb * factor.memory_factor
        assert 120 < adj_mem < 200, f"Memory {adj_mem:.1f} GB outside expected range"

    def test_mixtral_8x7b_memory(self):
        """Mixtral 8x7B FP16: actual memory ~90 GB"""
        params = self.analyzer.create_params(
            model_type="moe", preset_name="Mixtral 8x7B",
            quant="FP16", concurrency=1, max_tokens=2048,
        )
        result = self.engine.estimate(params)
        factor = self.cal_mgr.get_factor("moe", "NVIDIA A100 80GB")
        adj_mem = result.memory_gb * factor.memory_factor
        assert 70 < adj_mem < 120, f"Memory {adj_mem:.1f} GB outside expected range"

    def test_llama3_8b_gqa_less_kv_cache(self):
        """LLaMA-3 8B (GQA) should have less KV cache than LLaMA-2 7B (MHA)"""
        p_llama3 = self.analyzer.create_params(
            model_type="dense", preset_name="LLaMA-3 8B",
            quant="FP16", concurrency=1, max_tokens=2048,
        )
        p_llama2 = self.analyzer.create_params(
            model_type="dense", preset_name="LLaMA-2 7B",
            quant="FP16", concurrency=1, max_tokens=2048,
        )
        r_llama3 = self.engine.estimate(p_llama3)
        r_llama2 = self.engine.estimate(p_llama2)
        assert r_llama3.kv_cache_gb < r_llama2.kv_cache_gb
