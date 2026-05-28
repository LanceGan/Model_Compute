import streamlit as st
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager
from python.web.components.charts import render_throughput_bar

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

    st.header("多硬件对比")
    st.markdown("输入模型参数，同时对比所有可用硬件的性能表现。")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("负载类型", ["dense", "moe", "o1_reasoning", "multimodal", "recommendation"],
                                  format_func=lambda x: {"dense": "稠密", "moe": "MoE", "o1_reasoning": "类o1", "multimodal": "多模态", "recommendation": "生成式推荐模型"}[x], key="cmp_type")
        presets = analyzer.list_presets()
        preset_name = st.selectbox("模型预设", ["自定义"] + presets.get(model_type, []), key="cmp_preset")
        if preset_name == "自定义":
            if model_type == "recommendation":
                param_billions = st.number_input("骨干参数量 (B)（序列推荐用，DLRM填0）", min_value=0.0, value=0.0, step=0.1, key="cmp_params")
            else:
                param_billions = st.number_input("参数量 (B)", 0.1, value=7.0, key="cmp_params")
        else:
            param_billions = None
    with col2:
        quant = st.selectbox("量化方案", ["FP16", "INT8", "INT4"], key="cmp_quant")
        concurrency = st.slider("并发量", 1, 500, 1, key="cmp_conc")
        max_tokens = st.slider("最大 tokens", 512, 32768, 2048, step=512, key="cmp_tokens")

    # Conditional params
    reasoning_depth = 0
    image_resolution = 0
    num_images = 1

    if model_type == "o1_reasoning":
        reasoning_depth = st.selectbox(
            "推理深度", [1, 2, 3],
            format_func=lambda x: {1: "轻度 (2-3x)", 2: "中度 (3-5x)", 3: "重度 (5-10x)"}[x],
            key="cmp_depth",
        )

    if model_type == "multimodal":
        image_resolution = st.selectbox("图像分辨率", [224, 336, 448], index=1, key="cmp_res")
        num_images = st.slider("图像数量", 1, 10, 1, key="cmp_imgs")

    num_sparse_features = 0
    vocab_size_per_feature = 0
    embed_dim = 0
    mlp_dims = []

    if model_type == "recommendation":
        num_sparse_features = st.number_input("稀疏特征数", min_value=1, value=26, key="cmp_sparse")
        vocab_size_per_feature = st.number_input("每特征词表大小", min_value=1000, value=100000, step=10000, key="cmp_vocab")
        embed_dim = st.number_input("Embedding 维度", min_value=8, value=128, step=8, key="cmp_embed")
        mlp_str = st.text_input("MLP 层维度（逗号分隔，留空表示序列推荐）", "512,256,1", key="cmp_mlp")
        if mlp_str.strip():
            mlp_dims = [int(x.strip()) for x in mlp_str.split(",") if x.strip()]

    if st.button("开始对比", type="primary", use_container_width=True):
        params = analyzer.create_params(
            model_type=model_type,
            preset_name=preset_name if preset_name != "自定义" else None,
            param_billions=param_billions,
            quant=quant, concurrency=concurrency, max_tokens=max_tokens,
            reasoning_depth=reasoning_depth,
            image_resolution=image_resolution,
            num_images=num_images,
            num_sparse_features=num_sparse_features,
            vocab_size_per_feature=vocab_size_per_feature,
            embed_dim=embed_dim,
            mlp_dims=mlp_dims,
        )

        engine = mc.EstimationEngine()
        result = engine.estimate(params)
        matcher = mc.HardwareMatcher()
        hw_pool = hw_db.to_cpp_hardware_list()
        configs = matcher.match(result, params, hw_pool, 10.0)

        if not configs:
            st.warning("未找到匹配的硬件。")
            return

        # Results table
        rows = []
        for c in configs:
            factor = cal_mgr.get_factor(model_type, c.hardware.name)
            adj_tp = c.estimated_throughput * factor.throughput_factor
            rows.append({
                "硬件": c.hardware.name,
                "厂商": c.hardware.vendor,
                "卡数": c.num_cards,
                "单卡显存 (GB)": c.hardware.memory_gb,
                "预估吞吐 (tokens/s)": round(adj_tp, 1),
                "预估延迟 (ms)": round(c.estimated_latency_ms, 1),
                "瓶颈": c.bottleneck_type,
                "并行策略": c.parallel_strategy,
                "满足基线": c.meets_baseline,
            })

        st.dataframe(rows, use_container_width=True)

        # Chart
        chart_data = [{"hardware": r["硬件"], "throughput": r["预估吞吐 (tokens/s)"], "meets_baseline": r["满足基线"]} for r in rows]
        fig = render_throughput_bar(chart_data)
        st.plotly_chart(fig, use_container_width=True)
