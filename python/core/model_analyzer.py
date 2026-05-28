import json
import os
import sys
from pathlib import Path
from typing import Optional

_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

# On Windows, add MinGW DLL directories so the pyd can find its dependencies
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for candidate in [
        Path("F:/msys64/ucrt64/bin"),
        Path("C:/msys64/ucrt64/bin"),
        Path.home() / "msys64" / "ucrt64" / "bin",
    ]:
        if candidate.exists():
            os.add_dll_directory(str(candidate))
            break

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
    "recommendation": _mc.ModelType.RECOMMENDATION if _mc else None,
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
        reasoning_depth: Optional[int] = None,
        image_resolution: Optional[int] = None,
        num_images: int = 1,
        num_sparse_features: int = 0,
        vocab_size_per_feature: int = 0,
        embed_dim: int = 0,
        mlp_dims: Optional[list] = None,
    ):
        if mlp_dims is None:
            mlp_dims = []
        if _mc is None:
            raise RuntimeError("C++ module not available. Build with: bash scripts/build.sh")

        if model_type not in _TYPE_MAP:
            raise ValueError(f"未知的模型类型: {model_type}，可选: {list(_TYPE_MAP.keys())}")
        if quant not in _QUANT_MAP:
            raise ValueError(f"未知的量化方案: {quant}，可选: {list(_QUANT_MAP.keys())}")

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
            params.num_sparse_features = preset.get("num_sparse_features", 0)
            params.vocab_size_per_feature = preset.get("vocab_size_per_feature", 0)
            params.embed_dim = preset.get("embed_dim", 0)
            params.mlp_dims = preset.get("mlp_dims", [])
        elif param_billions is not None:
            params.param_billions = param_billions
            params.num_experts = num_experts
            params.active_experts = active_experts
            params.reasoning_depth = reasoning_depth if reasoning_depth is not None else 0
            params.image_resolution = image_resolution if image_resolution is not None else 0
            params.num_sparse_features = num_sparse_features
            params.vocab_size_per_feature = vocab_size_per_feature
            params.embed_dim = embed_dim
            params.mlp_dims = mlp_dims

        # Override if explicitly provided (including 0, which disables the feature)
        if reasoning_depth is not None:
            params.reasoning_depth = reasoning_depth
        if image_resolution is not None:
            params.image_resolution = image_resolution
        params.num_images = num_images

        return params
