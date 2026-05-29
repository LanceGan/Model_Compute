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

        # Auto-load default calibration data
        default_csv = self._dir / "default_calibration.csv"
        if default_csv.exists():
            self.import_csv(str(default_csv))
            for p in self._points:
                p["_is_default"] = True

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

        valid_tp = [p["actual_throughput"] / p["predicted_throughput"]
                     for p in matching if p["predicted_throughput"] > 0 and p["actual_throughput"] > 0]
        valid_mem = [p["actual_memory"] / p["predicted_memory"]
                     for p in matching if p["predicted_memory"] > 0 and p["actual_memory"] > 0]
        avg_tp = sum(valid_tp) / len(valid_tp) if valid_tp else 1.0
        avg_mem = sum(valid_mem) / len(valid_mem) if valid_mem else 1.0
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
        user_points = [p for p in self._points if not p.get("_is_default", False)]
        with open(save_path, "w") as f:
            f.write("# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem\n")
            for p in user_points:
                f.write(f"{p['model_type']},{p['hardware_name']},"
                        f"{p['predicted_throughput']},{p['actual_throughput']},"
                        f"{p['predicted_memory']},{p['actual_memory']}\n")

    def load(self, path: Optional[str] = None):
        self._points.clear()
        if self._cal:
            self._cal = _mc.Calibration()
        # Always reload defaults first
        default_csv = self._dir / "default_calibration.csv"
        if default_csv.exists():
            self.import_csv(str(default_csv))
        # Then load user calibration
        load_path = Path(path) if path else self._dir / "calibration.csv"
        if load_path.exists():
            self.import_csv(str(load_path))

    def list_entries(self) -> list[dict]:
        return self._points

    def adjust_throughput(self, predicted: float, model_type: str, hardware_name: str) -> float:
        factor = self.get_factor(model_type, hardware_name)
        return predicted * factor.throughput_factor

    def adjust_memory(self, predicted: float, model_type: str, hardware_name: str) -> float:
        factor = self.get_factor(model_type, hardware_name)
        return predicted * factor.memory_factor
