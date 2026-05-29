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
    initial_count = len(mgr.list_entries())  # includes auto-loaded defaults
    mgr.add_point("dense", "A100", 20.0, 16.0, 14.0, 15.0)
    mgr.add_point("moe", "H100", 30.0, 27.0, 50.0, 55.0)
    entries = mgr.list_entries()
    assert len(entries) == initial_count + 2


def test_default_calibration_auto_loaded():
    mgr = CalibrationManager()
    factor = mgr.get_factor("dense", "NVIDIA A100 80GB")
    assert factor.num_points > 0
    assert abs(factor.throughput_factor - 0.682) < 0.01  # 15/22 = 0.6818
    assert abs(factor.memory_factor - 1.107) < 0.01     # 15.5/14 = 1.107


def test_default_calibration_count():
    mgr = CalibrationManager()
    entries = mgr.list_entries()
    # Should have 25+ entries from the expanded default data
    assert len(entries) >= 25
