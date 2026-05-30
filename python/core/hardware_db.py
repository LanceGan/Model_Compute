import json
import os
from pathlib import Path
from typing import Optional

# Import C++ bindings
import sys
_build_path = Path(__file__).parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

# On Windows, add MinGW DLL directories so the pyd can find its dependencies
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Try MSYS2_PREFIX env var first, then search PATH and common locations
    _dll_candidates = []
    _msys_prefix = os.environ.get("MSYS2_PREFIX")
    if _msys_prefix:
        _dll_candidates.append(Path(_msys_prefix) / "bin")
    _dll_candidates.append(Path.home() / "msys64" / "ucrt64" / "bin")
    for _p in os.environ.get("PATH", "").split(os.pathsep):
        if "msys64" in _p.lower() and "ucrt64" in _p.lower():
            _dll_candidates.append(Path(_p))
    for _candidate in _dll_candidates:
        if _candidate.exists():
            os.add_dll_directory(str(_candidate))
            break

try:
    import model_compute as _mc
except ImportError:
    _mc = None


_DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "hardware_specs.json"


class HardwareDB:
    """Hardware specification database.

    Manages hardware specs (GPU/NPU) loaded from JSON, supports
    CRUD operations and conversion to C++ HardwareSpec objects
    for the estimation engine.
    """

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
