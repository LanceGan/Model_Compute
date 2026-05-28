import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render_throughput_bar(configs: list[dict]) -> go.Figure:
    """Render bar chart comparing throughput across hardware options."""
    df = pd.DataFrame(configs)
    fig = px.bar(
        df, x="hardware", y="throughput", color="meets_baseline",
        title="预估吞吐量对比 (tokens/s)",
        labels={"hardware": "硬件型号", "throughput": "吞吐量", "meets_baseline": "满足基线"},
        color_discrete_map={True: "#2ed573", False: "#ff4757"},
    )
    fig.add_hline(y=10, line_dash="dash", line_color="yellow", annotation_text="基线 10 tokens/s")
    return fig


def render_memory_breakdown(result) -> go.Figure:
    """Render pie chart of memory breakdown."""
    labels = ["模型权重", "KV Cache", "其他"]
    values = [result.weight_memory_gb, result.kv_cache_gb,
              max(0, result.memory_gb - result.weight_memory_gb - result.kv_cache_gb)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="显存占用分布")
    return fig


def render_sensitivity_curve(x_values, y_values, x_label, y_label, title) -> go.Figure:
    """Render line chart for sensitivity analysis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=y_label))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig
