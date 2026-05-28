import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager
from python.web.components.charts import render_throughput_bar, render_memory_breakdown

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

    st.header("算力估算")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("输入参数")

        model_type = st.selectbox(
            "负载类型",
            ["dense", "moe", "o1_reasoning", "multimodal"],
            format_func=lambda x: {
                "dense": "稠密模型 (Dense)",
                "moe": "MoE 模型",
                "o1_reasoning": "类o1推理模型",
                "multimodal": "多模态模型",
            }[x],
        )

        # Preset selection
        presets = analyzer.list_presets()
        preset_names = presets.get(model_type, [])
        preset_name = st.selectbox("模型预设", ["自定义"] + preset_names)

        if preset_name == "自定义":
            param_billions = st.number_input("参数量 (B)", min_value=0.1, value=7.0, step=0.1)
        else:
            param_billions = None

        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"])
        concurrency = st.slider("并发量", min_value=1, max_value=500, value=1)
        max_tokens = st.slider("最大 tokens 数", min_value=512, max_value=32768, value=2048, step=512)

        # Conditional params
        reasoning_depth = 0
        image_resolution = 0
        num_images = 1

        if model_type == "o1_reasoning":
            reasoning_depth = st.selectbox(
                "推理深度", [1, 2, 3],
                format_func=lambda x: {1: "轻度 (2-3x)", 2: "中度 (3-5x)", 3: "重度 (5-10x)"}[x],
            )

        if model_type == "multimodal":
            image_resolution = st.selectbox("图像分辨率", [224, 336, 448], index=1)
            num_images = st.slider("图像数量", 1, 10, 1)

        run_button = st.button("开始估算", type="primary", use_container_width=True)

    if run_button:
        with col_result:
            st.subheader("估算结果")

            # Create model params
            params = analyzer.create_params(
                model_type=model_type,
                preset_name=preset_name if preset_name != "自定义" else None,
                param_billions=param_billions,
                quant=quant,
                concurrency=concurrency,
                max_tokens=max_tokens,
                reasoning_depth=reasoning_depth,
                image_resolution=image_resolution,
                num_images=num_images,
            )

            # Run estimation
            engine = mc.EstimationEngine()
            result = engine.estimate(params)

            # Display estimation results
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总显存需求", f"{result.memory_gb:.1f} GB")
            m2.metric("权重显存", f"{result.weight_memory_gb:.1f} GB")
            m3.metric("KV Cache", f"{result.kv_cache_gb:.1f} GB")
            m4.metric("总 FLOPs", f"{result.flops_total:.2e}")

            # Memory breakdown chart
            fig_mem = render_memory_breakdown(result)
            st.plotly_chart(fig_mem, use_container_width=True)

            # Hardware matching
            st.subheader("硬件推荐")
            matcher = mc.HardwareMatcher()
            hw_pool = hw_db.to_cpp_hardware_list()
            configs = matcher.match(result, params, hw_pool, baseline_throughput=10.0)

            if configs:
                rows = []
                for c in configs:
                    # Apply calibration via factor
                    factor = cal_mgr.get_factor(model_type, c.hardware.name)
                    adj_tp = c.estimated_throughput * factor.throughput_factor
                    adj_mem = result.memory_gb * factor.memory_factor

                    rows.append({
                        "硬件型号": c.hardware.name,
                        "卡数": c.num_cards,
                        "单卡显存": f"{c.hardware.memory_gb} GB",
                        "预估吞吐": f"{adj_tp:.1f} tokens/s",
                        "预估延迟": f"{c.estimated_latency_ms:.1f} ms",
                        "瓶颈类型": c.bottleneck_type,
                        "并行策略": c.parallel_strategy,
                        "满足基线": "✅" if c.meets_baseline else "❌",
                    })

                st.table(rows)

                # Throughput comparison chart
                chart_data = [
                    {
                        "hardware": r["硬件型号"],
                        "throughput": float(r["预估吞吐"].split()[0]),
                        "meets_baseline": "✅" in r["满足基线"],
                    }
                    for r in rows
                ]
                fig_tp = render_throughput_bar(chart_data)
                st.plotly_chart(fig_tp, use_container_width=True)
            else:
                st.warning("未找到匹配的硬件配置。")
