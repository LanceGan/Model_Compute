# 校准指南

## 为什么需要校准？

理论公式可能与实际性能存在系统性偏差。校准模块通过少量实测数据修正这些偏差。

## 校准流程

1. 在目标硬件上运行模型推理
2. 记录实际吞吐量和显存占用
3. 与工具的预测值对比
4. 导入校准数据
5. 工具自动应用校准系数

## 校准数据格式 (CSV)

```csv
# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem
dense,NVIDIA A100 80GB,20.0,16.0,14.0,15.5
moe,NVIDIA H100 80GB,50.0,42.0,60.0,65.0
```

## 导入校准数据

1. Web UI → 管理 → 校准数据 → 上传 CSV
2. 或使用 Python API:
   ```python
   from python.core.calibration_mgr import CalibrationManager
   mgr = CalibrationManager()
   mgr.import_csv("path/to/calibration.csv")
   mgr.save()
   ```

## 注意事项

- 校准系数按 (模型类型, 硬件类型) 分组
- 未校准的配置使用默认系数 1.0
- 建议每个组合至少 3 个数据点
