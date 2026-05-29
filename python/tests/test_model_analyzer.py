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


def test_create_dlrm_params():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="recommendation",
        preset_name="DLRM-small",
        quant="FP16",
        concurrency=1,
        max_tokens=2048,
    )
    assert params.num_sparse_features == 26
    assert params.vocab_size_per_feature == 100000
    assert params.embed_dim == 128
    assert len(params.mlp_dims) == 3

def test_create_sasrec_params():
    analyzer = ModelAnalyzer()
    params = analyzer.create_params(
        model_type="recommendation",
        preset_name="SASRec",
        quant="FP16",
        concurrency=1,
        max_tokens=2048,
    )
    assert params.num_sparse_features == 1
    assert params.embed_dim == 64
    assert len(params.mlp_dims) == 0

def test_recommendation_presets_listed():
    analyzer = ModelAnalyzer()
    presets = analyzer.list_presets()
    assert "recommendation" in presets
    assert len(presets["recommendation"]) >= 4


def test_llama3_preset_has_gqa():
    analyzer = ModelAnalyzer()
    preset = analyzer.get_preset("dense", "LLaMA-3 8B")
    assert preset is not None
    assert preset["num_kv_heads"] == 8
    assert preset["head_dim"] == 128
    assert preset["use_swiglu"] is True


def test_qwen25_preset():
    analyzer = ModelAnalyzer()
    preset = analyzer.get_preset("dense", "Qwen-2.5 7B")
    assert preset is not None
    assert preset["num_kv_heads"] == 4


def test_deepseek_v3_preset():
    analyzer = ModelAnalyzer()
    preset = analyzer.get_preset("moe", "DeepSeek-V3")
    assert preset is not None
    assert preset["num_experts"] == 256
    assert preset["active_experts"] == 8
