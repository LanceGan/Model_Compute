import streamlit as st
import pandas as pd
from python.core.model_analyzer import ModelAnalyzer
from python.core.hardware_db import HardwareDB
from python.core.calibration_mgr import CalibrationManager

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

    st.header("批量估算")
    st.markdown("上传 CSV 文件或手动输入多组配置，批量计算算力需求并导出结果。")

    # CSV template
    with st.expander("CSV 格式说明"):
        st.markdown("CSV 文件需包含以下列：")
        st.code("model_type,preset_name,quant,concurrency,max_tokens\ndense,LLaMA-2 7B,FP16,1,2048\ndense,LLaMA-2 70B,INT8,4,4096\nmoe,Mixtral 8x7B,FP16,1,2048", language="csv")
        st.markdown("- `model_type`: dense/moe/o1_reasoning/multimodal/recommendation")
        st.markdown("- `preset_name`: 预设名称（可选，留空使用自定义参数）")
        st.markdown("- `quant`: FP16/INT8/INT4")
        st.markdown("- `concurrency`: 并发量")
        st.markdown("- `max_tokens`: 最大token数")

    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

    # Manual input
    st.markdown("---")
    st.subheader("手动添加配置")

    configs = []
    num_configs = st.number_input("配置数量", 1, 10, 2)

    for i in range(num_configs):
        st.markdown(f"**配置 {i+1}**")
        cols = st.columns([2, 2, 1, 1, 1])
        with cols[0]:
            mt = st.selectbox("模型类型", ["dense", "moe", "o1_reasoning", "multimodal", "recommendation"], key=f"b_mt_{i}")
        with cols[1]:
            presets = analyzer.list_presets()
            preset = st.selectbox("预设", ["自定义"] + presets.get(mt, []), key=f"b_pre_{i}")
        with cols[2]:
            q = st.selectbox("量化", ["FP16", "INT8", "INT4"], key=f"b_q_{i}")
        with cols[3]:
            conc = st.number_input("并发", 1, 500, 1, key=f"b_conc_{i}")
        with cols[4]:
            tokens = st.number_input("tokens", 512, 32768, 2048, step=512, key=f"b_tok_{i}")

        if preset != "自定义":
            configs.append({"model_type": mt, "preset_name": preset, "quant": q, "concurrency": conc, "max_tokens": tokens})
        else:
            pb = st.number_input("参数量(B)", 0.1, value=7.0, key=f"b_pb_{i}")
            configs.append({"model_type": mt, "param_billions": pb, "quant": q, "concurrency": conc, "max_tokens": tokens})

    # Parse uploaded CSV
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            configs = []
            for _, row in df.iterrows():
                cfg = {
                    "model_type": row.get("model_type", "dense"),
                    "quant": row.get("quant", "FP16"),
                    "concurrency": int(row.get("concurrency", 1)),
                    "max_tokens": int(row.get("max_tokens", 2048)),
                }
                if "preset_name" in row and pd.notna(row["preset_name"]):
                    cfg["preset_name"] = str(row["preset_name"])
                elif "param_billions" in row and pd.notna(row["param_billions"]):
                    cfg["param_billions"] = float(row["param_billions"])
                configs.append(cfg)
            st.success(f"已加载 {len(configs)} 条配置")
        except Exception as e:
            st.error(f"CSV 解析失败: {e}")

    if st.button("开始批量估算", type="primary", use_container_width=True):
        engine = mc.EstimationEngine()
        matcher = mc.HardwareMatcher()
        hw_pool = hw_db.to_cpp_hardware_list()

        results = []
        for cfg in configs:
            try:
                params = analyzer.create_params(**cfg)
                result = engine.estimate(params)
                hw_configs = matcher.match(result, params, hw_pool, 10.0)
                best_hw = hw_configs[0] if hw_configs else None

                cal_type = cfg["model_type"]
                hw_name = best_hw.hardware.name if best_hw else ""
                factor = cal_mgr.get_factor(cal_type, hw_name)
                adj_mem = result.memory_gb * factor.memory_factor
                adj_tp = best_hw.estimated_throughput * factor.throughput_factor if best_hw else 0

                results.append({
                    "模型类型": cfg["model_type"],
                    "预设": cfg.get("preset_name", "自定义"),
                    "量化": cfg["quant"],
                    "并发": cfg["concurrency"],
                    "tokens": cfg["max_tokens"],
                    "显存(GB)": round(adj_mem, 1),
                    "FLOPs": f"{result.flops_total:.2e}",
                    "推荐硬件": best_hw.hardware.name if best_hw else "N/A",
                    "卡数": best_hw.num_cards if best_hw else 0,
                    "吞吐(tokens/s)": round(adj_tp, 1) if best_hw else 0,
                    "满足基线": "是" if (best_hw and best_hw.meets_baseline) else "否",
                })
            except Exception as e:
                results.append({
                    "模型类型": cfg.get("model_type", "?"),
                    "预设": cfg.get("preset_name", "自定义"),
                    "错误": str(e),
                })

        df_results = pd.DataFrame(results)
        st.subheader("批量估算结果")
        st.dataframe(df_results, use_container_width=True)

        # Download button
        csv = df_results.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载结果 CSV", csv, "estimation_results.csv", "text/csv")
