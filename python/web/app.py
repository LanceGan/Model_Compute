import streamlit as st

st.set_page_config(
    page_title="算力等效建模评估工具",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 模型对算力等效建模评估工具")
st.markdown("基于异构算力资源池，支持多负载的算力需求表征和硬件推荐。")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能",
    ["算力估算", "多硬件对比", "敏感性分析", "管理"],
    index=0,
)

if page == "算力估算":
    from python.web.pages.estimation import render
    render()
elif page == "多硬件对比":
    from python.web.pages.comparison import render
    render()
elif page == "敏感性分析":
    from python.web.pages.sensitivity import render
    render()
elif page == "管理":
    from python.web.pages.management import render
    render()
