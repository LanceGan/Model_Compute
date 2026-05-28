import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager
from python.web.components.charts import render_sensitivity_curve

import sys
from pathlib import Path

_build_path = Path(__file__).parent.parent.parent.parent / "build" / "cpp"
if _build_path.exists():
    sys.path.insert(0, str(_build_path))

try:
    import model_compute as mc
except ImportError:
    mc = None


def render():
    if mc is None:
        st.error("C++ 模块未编译。请先运行: `bash scripts/build.sh`")
        return

    analyzer = ModelAnalyzer()
    hw_db = HardwareDB()
    cal_mgr = CalibrationManager()
    cal_mgr.load()

    st.header("敏感性分析")
    st.markdown("分析不同参数变化对算力需求和预估性能的影响。")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("负载类型", ["dense", "moe", "o1_reasoning", "multimodal"],
                                  format_func=lambda x: {"dense": "稠密", "moe": "MoE", "o1_reasoning": "类o1", "multimodal": "多模态"}[x], key="sa_type")
        presets = analyzer.list_presets()
        preset_name = st.selectbox("模型预设", ["自定义"] + presets.get(model_type, []), key="sa_preset")
        if preset_name == "自定义":
            param_billions = st.number_input("参数量 (B)", 0.1, value=7.0, key="sa_params")
        else:
            param_billions = None
        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"], key="sa_quant")
    with col2:
        hardware_name = st.selectbox("目标硬件", [h["name"] for h in hw_db.list_hardware()], key="sa_hw")
        max_tokens = st.slider("最大 tokens", 512, 32768, 2048, step=512, key="sa_tokens")

    # Conditional params
    reasoning_depth = 0
    image_resolution = 0
    num_images = 1

    if model_type == "o1_reasoning":
        reasoning_depth = st.selectbox(
            "推理深度", [1, 2, 3],
            format_func=lambda x: {1: "轻度 (2-3x)", 2: "中度 (3-5x)", 3: "重度 (5-10x)"}[x],
            key="sa_depth",
        )

    if model_type == "multimodal":
        image_resolution = st.selectbox("图像分辨率", [224, 336, 448], index=1, key="sa_res")
        num_images = st.slider("图像数量", 1, 10, 1, key="sa_imgs")

    analysis_var = st.selectbox("分析变量", ["并发量", "参数量", "序列长度"], key="sa_var")

    if st.button("开始分析", type="primary", use_container_width=True):
        hw = hw_db.get_hardware(hardware_name)
        if not hw:
            st.error("硬件未找到")
            return

        hw_spec = hw_db.to_cpp_hardware_list()
        target_hw = [h for h in hw_spec if h.name == hardware_name]
        if not target_hw:
            st.error("硬件数据错误")
            return

        engine = mc.EstimationEngine()
        matcher = mc.HardwareMatcher()

        x_values = []
        y_throughput = []
        y_memory = []
        y_cards = []

        if analysis_var == "并发量":
            x_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
            for conc in x_range:
                params = analyzer.create_params(
                    model_type=model_type,
                    preset_name=preset_name if preset_name != "自定义" else None,
                    param_billions=param_billions,
                    quant=quant, concurrency=conc, max_tokens=max_tokens,
                    reasoning_depth=reasoning_depth,
                    image_resolution=image_resolution,
                    num_images=num_images,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    factor = cal_mgr.get_factor(model_type, hardware_name)
                    x_values.append(conc)
                    y_throughput.append(configs[0].estimated_throughput * factor.throughput_factor)
                    y_memory.append(result.memory_gb * factor.memory_factor)
                    y_cards.append(configs[0].num_cards)
            x_label = "并发量"

        elif analysis_var == "参数量":
            x_range = [1.0, 3.0, 7.0, 13.0, 30.0, 70.0, 130.0]
            for pb in x_range:
                params = analyzer.create_params(
                    model_type=model_type, param_billions=pb,
                    quant=quant, concurrency=1, max_tokens=max_tokens,
                    reasoning_depth=reasoning_depth,
                    image_resolution=image_resolution,
                    num_images=num_images,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    factor = cal_mgr.get_factor(model_type, hardware_name)
                    x_values.append(pb)
                    y_throughput.append(configs[0].estimated_throughput * factor.throughput_factor)
                    y_memory.append(result.memory_gb * factor.memory_factor)
                    y_cards.append(configs[0].num_cards)
            x_label = "参数量 (B)"

        else:  # 序列长度
            x_range = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            for seq in x_range:
                params = analyzer.create_params(
                    model_type=model_type,
                    preset_name=preset_name if preset_name != "自定义" else None,
                    param_billions=param_billions,
                    quant=quant, concurrency=1, max_tokens=seq,
                    reasoning_depth=reasoning_depth,
                    image_resolution=image_resolution,
                    num_images=num_images,
                )
                result = engine.estimate(params)
                configs = matcher.match(result, params, target_hw, 10.0)
                if configs:
                    factor = cal_mgr.get_factor(model_type, hardware_name)
                    x_values.append(seq)
                    y_throughput.append(configs[0].estimated_throughput * factor.throughput_factor)
                    y_memory.append(result.memory_gb * factor.memory_factor)
                    y_cards.append(configs[0].num_cards)
            x_label = "最大序列长度"

        if x_values:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig1 = render_sensitivity_curve(x_values, y_throughput, x_label, "吞吐量 (tokens/s)", "吞吐量变化")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = render_sensitivity_curve(x_values, y_memory, x_label, "显存 (GB)", "显存需求变化")
                st.plotly_chart(fig2, use_container_width=True)
            with c3:
                fig3 = render_sensitivity_curve(x_values, y_cards, x_label, "卡数", "卡数变化")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("未找到匹配的硬件配置。")
