import pytest
import json
import tempfile
import os
from python.core.hardware_db import HardwareDB


def test_load_default_hardware():
    db = HardwareDB()
    hw_list = db.list_hardware()
    assert len(hw_list) >= 9
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
