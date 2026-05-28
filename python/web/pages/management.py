import streamlit as st
import pandas as pd
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager


def render():
    hw_db = HardwareDB()
    cal_mgr = CalibrationManager()
    cal_mgr.load()

    st.header("系统管理")

    tab_hw, tab_cal, tab_about = st.tabs(["硬件数据库", "校准数据", "关于"])

    with tab_hw:
        st.subheader("硬件数据库管理")

        # List existing hardware
        hw_list = hw_db.list_hardware()
        if hw_list:
            df = pd.DataFrame([
                {
                    "名称": h["name"],
                    "厂商": h["vendor"],
                    "类型": h["type"],
                    "FP16 TFLOPS": h["specs"]["fp16_tflops"],
                    "显存 (GB)": h["specs"]["memory_gb"],
                    "带宽 (GB/s)": h["specs"]["memory_bandwidth_gbs"],
                }
                for h in hw_list
            ])
            st.dataframe(df, use_container_width=True)

        # Add new hardware
        st.markdown("---")
        st.subheader("添加新硬件")
        with st.form("add_hardware"):
            name = st.text_input("硬件名称")
            vendor = st.text_input("厂商")
            hw_type = st.selectbox("类型", ["GPU", "NPU"])
            col1, col2, col3 = st.columns(3)
            with col1:
                fp16 = st.number_input("FP16 TFLOPS", 0.0, value=100.0)
                mem = st.number_input("显存 (GB)", 0.0, value=32.0)
            with col2:
                int8 = st.number_input("INT8 TOPS", 0.0, value=200.0)
                bw = st.number_input("带宽 (GB/s)", 0.0, value=1000.0)
            with col3:
                nvlink = st.number_input("NVLink 带宽 (GB/s)", 0.0, value=0.0)
                cost = st.number_input("参考价格", 0.0, value=5000.0)

            if st.form_submit_button("添加"):
                hw = {
                    "name": name, "vendor": vendor, "architecture": "", "type": hw_type,
                    "specs": {
                        "fp16_tflops": fp16, "int8_tops": int8, "fp32_tflops": 0,
                        "memory_gb": mem, "memory_type": "", "memory_bandwidth_gbs": bw,
                        "nvlink_bandwidth_gbs": nvlink, "pcie_version": "", "max_tdp_watts": 0,
                    },
                    "cost_per_unit": cost, "notes": "",
                }
                hw_db.add_hardware(hw)
                hw_db.save()
                st.success(f"已添加: {name}")
                st.rerun()

        # Remove hardware
        st.markdown("---")
        st.subheader("删除硬件")
        del_name = st.selectbox("选择要删除的硬件", [h["name"] for h in hw_list])
        if st.button("删除", type="secondary"):
            hw_db.remove_hardware(del_name)
            hw_db.save()
            st.success(f"已删除: {del_name}")
            st.rerun()

    with tab_cal:
        st.subheader("校准数据管理")

        entries = cal_mgr.list_entries()
        if entries:
            df = pd.DataFrame(entries)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无校准数据。可以通过 CSV 文件导入或在下方手动添加。")

        # Import CSV
        st.markdown("---")
        st.subheader("导入校准数据 (CSV)")
        uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
        if uploaded:
            import tempfile, os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                    f.write(uploaded.getvalue().decode())
                    tmp_path = f.name
                count = cal_mgr.import_csv(tmp_path)
                cal_mgr.save()
                st.success(f"成功导入 {count} 条校准记录")
                st.rerun()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Manual add
        st.markdown("---")
        st.subheader("手动添加校准记录")
        with st.form("add_calibration"):
            col1, col2 = st.columns(2)
            with col1:
                mt = st.selectbox("模型类型", ["dense", "moe", "o1_reasoning", "multimodal"], key="cal_mt")
                hw_name = st.text_input("硬件名称", key="cal_hw")
            with col2:
                pred_tp = st.number_input("预测吞吐", 0.0, value=20.0, key="cal_pred_tp")
                act_tp = st.number_input("实际吞吐", 0.0, value=16.0, key="cal_act_tp")
                pred_mem = st.number_input("预测显存", 0.0, value=14.0, key="cal_pred_mem")
                act_mem = st.number_input("实际显存", 0.0, value=15.0, key="cal_act_mem")
            if st.form_submit_button("添加"):
                cal_mgr.add_point(mt, hw_name, pred_tp, act_tp, pred_mem, act_mem)
                cal_mgr.save()
                st.success("校准记录已添加")
                st.rerun()

    with tab_about:
        st.subheader("关于本工具")
        st.markdown("""
        **模型对算力等效建模评估工具 v1.0**

        支持的模型类型：
        - 稠密模型 (Dense): LLaMA, GPT, Qwen 等
        - MoE 模型: Mixtral, DeepSeek-V2 等
        - 类o1推理模型: DeepSeek-R1 等
        - 多模态模型: LLaVA, Qwen-VL 等

        支持的硬件：
        - NVIDIA GPU: A100, H100, H200, L40S, RTX 4090
        - 昇腾 NPU: 910B, 300I Duo
        - 其他国产: 寒武纪 MLU370

        **架构:** C++ 核心引擎 + Python 业务层 + Streamlit Web UI
        """)
