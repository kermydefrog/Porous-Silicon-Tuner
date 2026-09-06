"""
Porous Silicon Tuner — Streamlit GUI
Trains RF + GB + Ridge + MLP on startup, then lets users
predict porosity (%) and thickness (nm) from etching parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import joblib
from ml_pipeline import load_data, FEATURES, TSV_PATH

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILE = {
    "Ridge":             "ridge",
    "Random Forest":     "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "MLP":               "mlp",
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Porous Silicon Tuner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8FBF9; }
[data-testid="stSidebar"] { background: #1B5E42; }

/* Sidebar headings and labels */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #E8F5E9 !important; }

/* Slider value label */
[data-testid="stSidebar"] [data-testid="stSlider"] p { color: #E8F5E9 !important; }

/* Selectbox and number inputs — keep text dark so value is readable */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div,
[data-testid="stSidebar"] input { color: #1A2420 !important; background: white !important; }

/* Dividers */
[data-testid="stSidebar"] hr { border-color: #2D6A4F; }

html, body, [class*="css"] { font-size: 18px !important; }
h1 { color: #1B5E42; }
h2, h3 { color: #2D6A4F; }
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #1B5E42;
    margin-bottom: 12px;
}
.metric-value { font-size: 2.4rem; font-weight: 700; color: #1B5E42; line-height: 1; }
.metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
.result-box {
    background: linear-gradient(135deg, #1B5E42 0%, #2D6A4F 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    text-align: center;
    margin: 8px 0;
}
.result-number { font-size: 3rem; font-weight: 800; letter-spacing: -1px; }
.result-unit { font-size: 1.1rem; opacity: 0.8; margin-top: 4px; }
.stButton > button {
    background: #1B5E42;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2D6A4F; }
</style>
""", unsafe_allow_html=True)

# ── Model training (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    df = load_data(TSV_PATH)
    models = {}
    for target in ["porosity", "thickness"]:
        models[target] = {}
        for display_name, file_key in MODEL_FILE.items():
            path = MODELS_DIR / f"{target}_{file_key}.joblib"
            models[target][display_name] = joblib.load(path)
    return df, models

df, models = load_models()

# ── Sidebar — inputs ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Etching Parameters")
    st.markdown("---")

    hf_ratio = st.selectbox(
        "HF : Ethanol ratio",
        options=["3:1", "1:1", "1:3", "3:2:1"],
        index=0,
        help="Volumetric ratio of HF to ethanol in the electrolyte"
    )

    current_density = st.slider(
        "Current density (mA/cm²)",
        min_value=5, max_value=400, value=100, step=1,
    )

    etch_time = st.slider(
        "Etch time (s)",
        min_value=30, max_value=600, value=180, step=10,
    )

    resistivity = st.selectbox(
        "Wafer resistivity (Ω·cm)",
        options=[1.00, 1.09, 1.20],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Sacrificial Layer")

    sac_current = st.selectbox(
        "Sacrificial current (mA/cm²)",
        options=[50, 100, 200],
        index=2,
    )

    sac_time = st.selectbox(
        "Sacrificial time (s)",
        options=[30, 120, 240],
        index=0,
    )

    st.markdown("---")
    model_choice = st.selectbox(
        "Prediction model",
        options=["MLP", "Ridge", "Gradient Boosting", "Random Forest"],
        index=0,
    )


# ── Main layout ────────────────────────────────────────────────────────────────

st.title("Porous Silicon Tuner")
st.markdown(
    "Predict **porosity (%)** and **layer thickness (nm)** from electrochemical "
    "etching parameters using machine learning models."
)

# ── Prediction ─────────────────────────────────────────────────────────────────

input_row = pd.DataFrame([{
    "HF": hf_ratio,
    "current_density": float(current_density),
    "etch_time": float(etch_time),
    "resistivity": float(resistivity),
    "sac_current": float(sac_current),
    "sac_time": float(sac_time),
}])

pipe_por = models["porosity"][model_choice]["pipe"]
pipe_thk = models["thickness"][model_choice]["pipe"]

pred_porosity  = round(float(pipe_por.predict(input_row)[0]), 1)
pred_thickness = round(float(pipe_thk.predict(input_row)[0]), 0)

# Clamp to physically plausible range
pred_porosity  = max(0.0, min(95.0, pred_porosity))
pred_thickness = max(0.0, pred_thickness)

# ── Results row ────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="result-box">
        <div class="result-unit">Porosity</div>
        <div class="result-number">{pred_porosity:.1f}<span style="font-size:1.4rem">%</span></div>
        <div class="result-unit" style="margin-top:8px">{model_choice}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-box" style="background: linear-gradient(135deg, #1565C0 0%, #1976D2 100%);">
        <div class="result-unit">Thickness</div>
        <div class="result-number">{int(pred_thickness):,}<span style="font-size:1.4rem">nm</span></div>
        <div class="result-unit" style="margin-top:8px">{model_choice}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    r2_por = models["porosity"][model_choice]["R2"]
    r2_thk = models["thickness"][model_choice]["R2"]
    mae_por = models["porosity"][model_choice]["MAE"]
    mae_thk = models["thickness"][model_choice]["MAE"]
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-weight:600; color:#1B5E42; margin-bottom:10px">Model accuracy (test set)</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px">
            <span style="color:#444">Porosity R²</span>
            <strong>{r2_por:.3f}</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px">
            <span style="color:#444">Porosity MAE</span>
            <strong>{mae_por:.2f}%</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px">
            <span style="color:#444">Thickness R²</span>
            <strong>{r2_thk:.3f}</strong>
        </div>
        <div style="display:flex; justify-content:space-between">
            <span style="color:#444">Thickness MAE</span>
            <strong>{int(mae_thk):,} nm</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Sweep", "🎯 Model Comparison", "📊 Dataset Explorer"])

# ── Tab 1: Sensitivity sweep ───────────────────────────────────────────────────

with tab1:
    st.markdown("### How does each parameter affect the predictions?")
    st.markdown("The selected parameter is swept across its full range; all others are fixed at your sidebar values.")

    sweep_param = st.selectbox(
        "Sweep parameter",
        options=["current_density", "etch_time", "sac_current", "sac_time"],
        format_func=lambda x: {
            "current_density": "Current density (mA/cm²)",
            "etch_time": "Etch time (s)",
            "sac_current": "Sacrificial current (mA/cm²)",
            "sac_time": "Sacrificial time (s)",
        }[x],
    )

    sweep_ranges = {
        "current_density": np.linspace(5, 400, 80),
        "etch_time":       np.linspace(30, 600, 80),
        "sac_current":     np.array([50, 100, 200]),
        "sac_time":        np.array([30, 120, 240]),
    }
    sweep_vals = sweep_ranges[sweep_param]

    base = {
        "HF": hf_ratio,
        "current_density": float(current_density),
        "etch_time": float(etch_time),
        "resistivity": float(resistivity),
        "sac_current": float(sac_current),
        "sac_time": float(sac_time),
    }
    rows = []
    for v in sweep_vals:
        r = dict(base)
        r[sweep_param] = float(v)
        rows.append(r)
    sweep_df = pd.DataFrame(rows)

    por_sweep = np.clip(pipe_por.predict(sweep_df), 0.0, 95.0)
    thk_sweep = np.clip(pipe_thk.predict(sweep_df), 0.0, None)

    x_label = {
        "current_density": "Current density (mA/cm²)",
        "etch_time": "Etch time (s)",
        "sac_current": "Sacrificial current (mA/cm²)",
        "sac_time": "Sacrificial time (s)",
    }[sweep_param]

    fig_sweep = make_subplots(rows=1, cols=2,
        subplot_titles=("Porosity (%)", "Thickness (nm)"))

    mode = "lines" if len(sweep_vals) > 5 else "lines+markers"

    fig_sweep.add_trace(
        go.Scatter(x=sweep_vals, y=por_sweep, mode=mode,
                   line=dict(color="#1B5E42", width=2.5),
                   marker=dict(size=8),
                   name="Porosity"),
        row=1, col=1
    )
    fig_sweep.add_trace(
        go.Scatter(x=[float(base[sweep_param])], y=[pred_porosity],
                   mode="markers",
                   marker=dict(color="#FF6F00", size=12, symbol="star"),
                   name="Current setting",
                   showlegend=True),
        row=1, col=1
    )
    fig_sweep.add_trace(
        go.Scatter(x=sweep_vals, y=thk_sweep, mode=mode,
                   line=dict(color="#1565C0", width=2.5),
                   marker=dict(size=8),
                   name="Thickness"),
        row=1, col=2
    )
    fig_sweep.add_trace(
        go.Scatter(x=[float(base[sweep_param])], y=[pred_thickness],
                   mode="markers",
                   marker=dict(color="#FF6F00", size=12, symbol="star"),
                   name="Current setting",
                   showlegend=False),
        row=1, col=2
    )

    fig_sweep.update_xaxes(
        title_text=x_label,
        title_font=dict(size=18),
        tickfont=dict(size=16),
    )
    fig_sweep.update_yaxes(title_text="Porosity (%)",   row=1, col=1,
                           title_font=dict(size=18), tickfont=dict(size=16))
    fig_sweep.update_yaxes(title_text="Thickness (nm)", row=1, col=2,
                           title_font=dict(size=18), tickfont=dict(size=16))
    fig_sweep.update_layout(
        height=460, template="plotly_white",
        legend=dict(orientation="h", y=-0.22, font=dict(size=15)),
        margin=dict(t=50, b=80),
        font=dict(size=16),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

# ── Tab 2: Model comparison ────────────────────────────────────────────────────

with tab2:
    st.markdown("### All models predict for your current parameter settings")

    model_names = ["Ridge", "Random Forest", "Gradient Boosting", "MLP"]
    por_preds, thk_preds = [], []
    for m in model_names:
        por_preds.append(round(float(models["porosity"][m]["pipe"].predict(input_row)[0]), 1))
        thk_preds.append(round(float(models["thickness"][m]["pipe"].predict(input_row)[0]), 0))

    colors = ["#9E9E9E", "#1B5E42", "#FF8F00", "#1565C0"]
    highlight = [1.0 if m == model_choice else 0.55 for m in model_names]

    fig_cmp = make_subplots(rows=1, cols=2,
        subplot_titles=("Porosity prediction (%)", "Thickness prediction (nm)"))

    fig_cmp.add_trace(
        go.Bar(x=model_names, y=por_preds,
               marker_color=colors,
               marker_opacity=highlight,
               text=[f"{v:.1f}%" for v in por_preds],
               textposition="outside", name="Porosity"),
        row=1, col=1
    )
    fig_cmp.add_trace(
        go.Bar(x=model_names, y=thk_preds,
               marker_color=colors,
               marker_opacity=highlight,
               text=[f"{int(v):,} nm" for v in thk_preds],
               textposition="outside", name="Thickness"),
        row=1, col=2
    )
    fig_cmp.update_layout(
        height=420, template="plotly_white",
        showlegend=False,
        margin=dict(t=40, b=20),
    )
    fig_cmp.update_yaxes(title_text="Porosity (%)", row=1, col=1)
    fig_cmp.update_yaxes(title_text="Thickness (nm)", row=1, col=2)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Actual vs predicted scatter for selected model
    st.markdown(f"### Actual vs. Predicted — {model_choice} (held-out test set)")
    c1, c2 = st.columns(2)

    for col_ui, target, color, unit in [
        (c1, "porosity",  "#1B5E42", "%"),
        (c2, "thickness", "#1565C0", "nm"),
    ]:
        res = models[target][model_choice]
        y_te   = res["y_te"].values
        y_pred = res["y_pred"]
        lo, hi = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_te, y=y_pred, mode="markers",
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            name="Test samples",
        ))
        fig_sc.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#E53935", dash="dash", width=1.5),
            name="Ideal",
        ))
        fig_sc.update_layout(
            height=340, template="plotly_white",
            xaxis_title=f"Actual {target} ({unit})",
            yaxis_title=f"Predicted {target} ({unit})",
            title=f"R²={res['R2']:.3f}  MAE={res['MAE']:.2f} {unit}",
            margin=dict(t=40, b=40, l=40, r=20),
            showlegend=False,
        )
        with col_ui:
            st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 3: Dataset explorer ────────────────────────────────────────────────────

with tab3:
    st.markdown("### Experimental dataset overview (n = 209)")

    hf_filter = st.multiselect(
        "Filter by HF ratio",
        options=df["HF"].dropna().unique().tolist(),
        default=df["HF"].dropna().unique().tolist(),
    )
    df_filtered = df[df["HF"].isin(hf_filter)] if hf_filter else df

    col_a, col_b = st.columns(2)
    color_map = {"1:3": "#E53935", "3:1": "#1E88E5", "1:1": "#43A047", "3:2:1": "#FB8C00"}

    with col_a:
        fig_por = go.Figure()
        for hf in hf_filter:
            sub = df_filtered[df_filtered["HF"] == hf]
            fig_por.add_trace(go.Scatter(
                x=sub["current_density"], y=sub["porosity"],
                mode="markers",
                marker=dict(color=color_map.get(hf, "#888"), size=7,
                            opacity=0.75, line=dict(color="white", width=0.5)),
                name=f"HF {hf}",
            ))
        # Mark current input
        fig_por.add_trace(go.Scatter(
            x=[current_density], y=[pred_porosity],
            mode="markers",
            marker=dict(color="#FF6F00", size=14, symbol="star",
                        line=dict(color="white", width=1.5)),
            name="Your prediction",
        ))
        fig_por.update_layout(
            height=360, template="plotly_white",
            xaxis_title="Current density (mA/cm²)",
            yaxis_title="Porosity (%)",
            title="Porosity vs. current density",
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_por, use_container_width=True)

    with col_b:
        fig_thk = go.Figure()
        for hf in hf_filter:
            sub = df_filtered[df_filtered["HF"] == hf]
            fig_thk.add_trace(go.Scatter(
                x=sub["etch_time"], y=sub["thickness"],
                mode="markers",
                marker=dict(color=color_map.get(hf, "#888"), size=7,
                            opacity=0.75, line=dict(color="white", width=0.5)),
                name=f"HF {hf}",
            ))
        fig_thk.add_trace(go.Scatter(
            x=[etch_time], y=[pred_thickness],
            mode="markers",
            marker=dict(color="#FF6F00", size=14, symbol="star",
                        line=dict(color="white", width=1.5)),
            name="Your prediction",
        ))
        fig_thk.update_layout(
            height=360, template="plotly_white",
            xaxis_title="Etch time (s)",
            yaxis_title="Thickness (nm)",
            title="Thickness vs. etch time",
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_thk, use_container_width=True)

    with st.expander("Show raw data"):
        st.dataframe(
            df_filtered[["HF", "current_density", "etch_time", "resistivity",
                          "sac_current", "sac_time", "porosity", "thickness"]
            ].rename(columns={
                "current_density": "J (mA/cm²)",
                "etch_time": "t_etch (s)",
                "resistivity": "ρ (Ω·cm)",
                "sac_current": "J_sac (mA/cm²)",
                "sac_time": "t_sac (s)",
                "porosity": "Porosity (%)",
                "thickness": "Thickness (nm)",
            }),
            use_container_width=True,
            height=320,
        )
