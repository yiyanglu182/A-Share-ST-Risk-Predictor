"""
A 股 ST 风险预警系统 — Streamlit Demo
运行方式: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── 页面基础设置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="A股ST风险预警",
    page_icon="⚠️",
    layout="wide",
)

# ── 加载模型和数据（只加载一次，缓存到内存）─────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "best_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DIR / "03_modeling_dataset.csv", encoding="utf-8-sig")
    results = pd.read_csv(MODELS_DIR / "model_results.csv", encoding="utf-8-sig", index_col=0)
    return df, results

bundle  = load_model()
model        = bundle["model"]
imputer      = bundle["imputer"]
scaler       = bundle["scaler"]
feature_cols = bundle["feature_cols"]
model_name   = bundle["model_name"]

df, results_df = load_data()

# 只看 2023 年测试集
test_df = df[df["year"] == 2023].copy().reset_index(drop=True)

# 预处理 + 预测
X_raw     = test_df[feature_cols]
X_imp     = imputer.transform(X_raw)
X_scaled  = scaler.transform(X_imp)
test_df["pred_prob"] = model.predict_proba(X_scaled)[:, 1]
test_df["pred_label"] = (test_df["pred_prob"] >= 0.5).astype(int)

# ── 标题区 ───────────────────────────────────────────────────────
st.title("⚠️ A 股 ST 风险预警系统")
st.markdown(
    "基于 **T-2 年财务数据**，提前 1-2 年识别上市公司被标记为 ST/*ST 的风险。  \n"
    f"当前模型：**{model_name}**"
)
st.divider()

# ── 第一行：模型指标 ──────────────────────────────────────────────
st.subheader("📊 模型评估结果（测试集：2023 年）")

col1, col2, col3, col4 = st.columns(4)
best_row = results_df.loc[model_name] if model_name in results_df.index else results_df.iloc[0]

col1.metric("AUC",       f"{best_row['AUC']:.3f}")
col2.metric("F1",        f"{best_row['F1']:.3f}")
col3.metric("Precision", f"{best_row['Precision']:.3f}")
col4.metric("Recall",    f"{best_row['Recall']:.3f}")

# ROC 曲线图
if (MODELS_DIR / "model_comparison.png").exists():
    with st.expander("查看 ROC 曲线对比图"):
        st.image(str(MODELS_DIR / "model_comparison.png"), use_container_width=True)

st.divider()

# ── 第二行：SHAP 特征重要性 ────────────────────────────────────────
st.subheader("🔍 Top 风险驱动特征（SHAP 分析）")

col_shap1, col_shap2 = st.columns([1, 1])

with col_shap1:
    if (MODELS_DIR / "shap_importance.png").exists():
        st.image(str(MODELS_DIR / "shap_importance.png"), use_container_width=True)

with col_shap2:
    if (MODELS_DIR / "shap_feature_ranking.csv").exists():
        ranking = pd.read_csv(MODELS_DIR / "shap_feature_ranking.csv",
                              encoding="utf-8-sig", index_col=0)
        st.markdown("**Top 10 ST 风险驱动指标**")
        st.dataframe(ranking.head(10), use_container_width=True)

st.divider()

# ── 第三行：单公司风险诊断 ────────────────────────────────────────
st.subheader("🏢 单公司风险诊断")

# 下拉菜单：选择公司
test_df["display"] = (
    test_df["stock_code"] + "  " +
    test_df["stock_name"] + "  " +
    test_df["label"].map({1: "🔴 实际ST", 0: "🟢 正常"})
)
selected = st.selectbox(
    "选择一家公司（2023 年测试集，共 142 家）：",
    options=test_df["display"].tolist(),
)

# 取选中的公司
idx = test_df[test_df["display"] == selected].index[0]
row = test_df.loc[idx]

# 风险评分展示
prob = row["pred_prob"]
actual = "🔴 实际被 ST" if row["label"] == 1 else "🟢 实际正常"

c1, c2, c3 = st.columns(3)
c1.metric("股票代码", row["stock_code"])
c2.metric("公司名称", row["stock_name"])
c3.metric("实际结果", actual)

# 风险概率进度条
risk_color = "🔴" if prob >= 0.6 else ("🟡" if prob >= 0.4 else "🟢")
st.markdown(f"### {risk_color} ST 风险概率：**{prob:.1%}**")
st.progress(float(prob))

if prob >= 0.6:
    st.error("高风险：模型判断该公司在未来 1-2 年内被标记 ST 的概率较高")
elif prob >= 0.4:
    st.warning("中等风险：需持续关注财务指标变化")
else:
    st.success("低风险：当前财务状况相对健康")

# ── SHAP 瀑布图（实时计算）──────────────────────────────────────
st.markdown("#### 风险归因分析（SHAP 瀑布图）")
st.caption("正值（红色）= 推高 ST 风险；负值（蓝色）= 降低 ST 风险")

with st.spinner("正在计算 SHAP 值..."):
    explainer  = shap.TreeExplainer(model)
    raw_shap   = explainer(X_scaled)

    # 处理二分类 3D 情况
    vals_all = raw_shap.values
    base_all = raw_shap.base_values
    if len(vals_all.shape) == 3:
        vals_all = vals_all[:, :, 1]
        base_all = base_all[:, 1]

    single_exp = shap.Explanation(
        values       = vals_all[idx],
        base_values  = float(base_all[idx]),
        data         = raw_shap.data[idx],
        feature_names= feature_cols,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(single_exp, max_display=12, show=False)
    plt.title(
        f"{row['stock_name']}（{row['stock_code']}） — ST 风险概率 {prob:.1%}",
        fontsize=12
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── 底部：财务指标详情 ────────────────────────────────────────────
with st.expander("查看该公司完整财务指标"):
    fin_data = row[feature_cols].to_frame(name="数值")
    fin_data.index.name = "财务指标"
    st.dataframe(fin_data, use_container_width=True)
