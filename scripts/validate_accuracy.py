#!/usr/bin/env python3
"""Calculate MAPE (Mean Absolute Percentage Error) from calibration data."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager

_build_path = project_root / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    print("ERROR: C++ module not built. Run: bash scripts/build.sh")
    sys.exit(1)


def main():
    analyzer = ModelAnalyzer()
    hw_db = HardwareDB()
    cal_mgr = CalibrationManager()
    engine = mc.EstimationEngine()

    entries = cal_mgr.list_entries()
    if not entries:
        print("No calibration data found.")
        return

    print(f"Loaded {len(entries)} calibration entries")
    print("-" * 60)

    # Hoist preset lookup before the loop to avoid repeated calls
    presets = analyzer.list_presets()

    mem_errors = []
    flops_entries = []
    for entry in entries:
        model_type = entry["model_type"]
        actual_mem = entry["actual_memory"]
        actual_tp = entry["actual_throughput"]

        preset_names = presets.get(model_type, [])
        if not preset_names:
            continue

        # NOTE: calibration data may use different quantization than FP16.
        # If the original measurements used e.g. INT8 or FP32, the memory
        # and throughput predictions here will differ accordingly.
        params = analyzer.create_params(
            model_type=model_type,
            preset_name=preset_names[0],
            quant="FP16",
            concurrency=1,
            max_tokens=2048,
        )
        result = engine.estimate(params)

        if actual_mem > 0:
            mem_error = abs(result.memory_gb - actual_mem) / actual_mem * 100
            mem_errors.append((model_type, result.memory_gb, actual_mem, mem_error))

        # Note: throughput depends on hardware and is estimated by hardware matcher
        # We report FLOPs as compute requirement indicator
        if result.flops_total > 0:
            flops_entries.append((model_type, result.flops_total))

    if mem_errors:
        mape = sum(e[3] for e in mem_errors) / len(mem_errors)
        print(f"Memory MAPE: {mape:.1f}% ({len(mem_errors)} samples)")
        print()
        for mt, pred, actual, err in mem_errors:
            print(f"  {mt:20s} pred={pred:8.1f} GB  actual={actual:8.1f} GB  error={err:5.1f}%")
    else:
        print("No matching presets found for memory calibration entries.")

    print()
    print("Note: Throughput depends on hardware specs and is estimated by the")
    print("hardware matcher, not the estimation engine directly.")
    print("-" * 60)


if __name__ == "__main__":
    main()
