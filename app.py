"""
Probabilistic Default Modelling for Loan Portfolios
====================================================
Statistics Project Dashboard — No model training, instant load.
All logistic regression coefficients are pre-computed analytically.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Probabilistic Default Modelling",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLASSMORPHIC CSS
# ─────────────────────────────────────────────
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root & Background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 30%, #091520 60%, #0a0e1a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10, 20, 40, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(99,179,237,0.15);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed !important;
}

/* ── Global text ── */
h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown {
    color: #e2e8f0 !important;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 20px;
    padding: 28px 30px;
    margin-bottom: 22px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(120deg, rgba(66,153,225,0.18) 0%, rgba(107,70,193,0.18) 50%, rgba(236,72,153,0.12) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 24px;
    padding: 40px 44px;
    margin-bottom: 30px;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    text-align: center;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #b794f4, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(226,232,240,0.7) !important;
    font-weight: 400;
    margin: 0;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.20);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.25s ease;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(99,179,237,0.45); }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.78rem;
    color: rgba(226,232,240,0.6) !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Colour accents ── */
.accent-blue  { color: #63b3ed !important; }
.accent-green { color: #68d391 !important; }
.accent-red   { color: #fc8181 !important; }
.accent-purple{ color: #b794f4 !important; }
.accent-yellow{ color: #f6e05e !important; }
.accent-pink  { color: #f687b3 !important; }

/* ── Formula box ── */
.formula-box {
    background: rgba(107,70,193,0.12);
    border: 1px solid rgba(183,148,244,0.30);
    border-radius: 14px;
    padding: 18px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #b794f4 !important;
    margin: 12px 0;
    text-align: center;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.risk-low    { background: rgba(72,187,120,0.20); border:1px solid rgba(72,187,120,0.50); color:#68d391!important; }
.risk-medium { background: rgba(246,224,94,0.15); border:1px solid rgba(246,224,94,0.45); color:#f6e05e!important; }
.risk-high   { background: rgba(252,129,129,0.15); border:1px solid rgba(252,129,129,0.45); color:#fc8181!important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99,179,237,0.12);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: rgba(226,232,240,0.55) !important;
    font-weight: 500;
    padding: 10px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
    border: 1px solid rgba(99,179,237,0.30) !important;
}

/* ── Slider / input widgets ── */
.stSlider > div { color: #e2e8f0 !important; }
.stNumberInput label, .stSelectbox label { color: #e2e8f0 !important; }

/* ── Section divider ── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #63b3ed !important;
    border-left: 4px solid #63b3ed;
    padding-left: 14px;
    margin: 28px 0 18px 0;
}

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* ── Metric delta ── */
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: #63b3ed !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: rgba(226,232,240,0.65) !important; }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)", color="#a0aec0"),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)", color="#a0aec0"),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="rgba(10,20,40,0.95)", bordercolor="rgba(99,179,237,0.3)", font_color="#e2e8f0"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,237,0.15)"),
)

# ─────────────────────────────────────────────
#  DATA GENERATION (INSTANT — NO TRAINING)
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=800, seed=42):
    rng = np.random.default_rng(seed)

    age              = rng.integers(22, 65, n).astype(float)
    income           = np.clip(rng.normal(60000, 20000, n), 18000, 150000)
    loan_amount      = np.clip(rng.normal(120000, 60000, n), 10000, 500000)
    credit_score     = np.clip(rng.normal(650, 80, n), 300, 850)
    employment_years = np.clip(rng.normal(8, 5, n), 0, 40)
    debt_to_income   = np.clip(rng.normal(0.35, 0.15, n), 0.05, 0.95)
    interest_rate    = np.clip(rng.normal(10.5, 3.5, n), 4.0, 24.0)
    loan_term        = rng.choice([12, 24, 36, 48, 60, 84], n).astype(float)
    num_credit_lines = rng.integers(1, 12, n).astype(float)

    # Pre-computed logistic regression formula (no training)
    # z = β₀ + β₁·credit_score + β₂·income + β₃·dti + β₄·employment
    z = (
        3.5
        - 0.008  * credit_score
        - 0.000012 * income
        + 3.2   * debt_to_income
        - 0.06  * employment_years
        + 0.04  * interest_rate
        - 0.002 * num_credit_lines
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = (prob_default > rng.uniform(0, 1, n)).astype(int)

    # Expected Loss components
    lgd = np.clip(rng.beta(2, 3, n), 0.1, 0.9)   # Loss Given Default
    ead = loan_amount * rng.uniform(0.8, 1.0, n)   # Exposure at Default
    el  = prob_default * lgd * ead                  # Expected Loss

    df = pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "debt_to_income": debt_to_income, "interest_rate": interest_rate,
        "loan_term": loan_term, "num_credit_lines": num_credit_lines,
        "prob_default": prob_default, "default": default,
        "lgd": lgd, "ead": ead, "expected_loss": el,
    })
    return df

# ─────────────────────────────────────────────
#  PRE-COMPUTED SIGMOID HELPER
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_pd(credit_score, income, dti, employment_years, interest_rate, num_credit_lines):
    """Analytical PD using pre-set coefficients."""
    z = (3.5
         - 0.008   * credit_score
         - 0.000012 * income
         + 3.2     * dti
         - 0.06    * employment_years
         + 0.04    * interest_rate
         - 0.002   * num_credit_lines)
    return float(sigmoid(z)), float(z)

# ─────────────────────────────────────────────
#  CONFUSION MATRIX (ANALYTICAL)
# ─────────────────────────────────────────────
def compute_metrics(df):
    pred = (df["prob_default"] >= 0.5).astype(int)
    actual = df["default"]
    tp = int(((pred == 1) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    acc = (tp + tn) / len(df)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df = generate_dataset()
metrics = compute_metrics(df)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-size:2.8rem;'>🏦</div>
        <div style='font-size:1.1rem; font-weight:700; color:#63b3ed; margin-top:6px;'>Loan Default Model</div>
        <div style='font-size:0.76rem; color:rgba(226,232,240,0.5); margin-top:3px;'>Probabilistic Risk Engine</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📊 Portfolio Filter")
    risk_threshold = st.slider("Default Probability Threshold", 0.1, 0.9, 0.5, 0.05)
    income_filter  = st.slider("Min Annual Income (₹)", 18000, 150000, 18000, 5000, format="₹%d")
    credit_filter  = st.slider("Min Credit Score", 300, 850, 300, 10)

    filtered_df = df[
        (df["income"] >= income_filter) &
        (df["credit_score"] >= credit_filter)
    ].copy()
    filtered_df["high_risk"] = filtered_df["prob_default"] >= risk_threshold

    st.divider()
    st.markdown(f"""
    <div class='glass-card' style='padding:16px;'>
        <div style='font-size:0.78rem;color:rgba(226,232,240,0.55);text-transform:uppercase;letter-spacing:.08em;'>Filtered Loans</div>
        <div style='font-size:1.9rem;font-weight:800;color:#63b3ed;'>{len(filtered_df):,}</div>
        <div style='font-size:0.78rem;color:rgba(226,232,240,0.55);margin-top:6px;'>High-Risk Loans</div>
        <div style='font-size:1.9rem;font-weight:800;color:#fc8181;'>{filtered_df["high_risk"].sum():,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎓 Model Info")
    st.markdown("""
    <div style='font-size:0.82rem; color:rgba(226,232,240,0.65); line-height:1.7;'>
    ✅ Pre-computed Logistic Regression<br>
    ✅ No training delay — instant<br>
    ✅ Sigmoid probability output<br>
    ✅ EL = PD × LGD × EAD<br>
    ✅ 800-loan synthetic portfolio
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏦 Probabilistic Default Modelling</div>
    <div class='hero-sub'>Loan Portfolio Risk Analytics &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp; Expected Loss Model &nbsp;|&nbsp; Statistics Project</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
total_loans   = len(df)
total_defaults = df["default"].sum()
avg_pd        = df["prob_default"].mean()
portfolio_el  = df["expected_loss"].sum()
avg_credit    = df["credit_score"].mean()
total_ead     = df["ead"].sum()
avg_lgd       = df["lgd"].mean()
model_acc     = metrics["acc"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-blue'>{total_loans:,}</div>
        <div class='kpi-label'>Total Loans</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-red'>{total_defaults:,} <span style='font-size:1rem;'>({avg_pd*100:.1f}%)</span></div>
        <div class='kpi-label'>Defaults / Avg PD</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-yellow'>₹{portfolio_el/1e6:.2f}M</div>
        <div class='kpi-label'>Portfolio Expected Loss</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-green'>{model_acc*100:.1f}%</div>
        <div class='kpi-label'>Model Accuracy</div>
    </div>""", unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-purple'>{avg_credit:.0f}</div>
        <div class='kpi-label'>Avg Credit Score</div>
    </div>""", unsafe_allow_html=True)
with col6:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-pink'>₹{total_ead/1e6:.1f}M</div>
        <div class='kpi-label'>Total Exposure (EAD)</div>
    </div>""", unsafe_allow_html=True)
with col7:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-yellow'>{avg_lgd*100:.1f}%</div>
        <div class='kpi-label'>Avg Loss Given Default</div>
    </div>""", unsafe_allow_html=True)
with col8:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value accent-green'>{metrics["f1"]*100:.1f}%</div>
        <div class='kpi-label'>F1 Score</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Sigmoid & Probability",
    "📊 Portfolio Analytics",
    "💰 Expected Loss Model",
    "🎯 Loan Predictor",
    "🧮 Model Metrics",
])

# ═══════════════════════════════════════════════
#  TAB 1 — SIGMOID & PROBABILITY
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Sigmoid Function — The Heart of Logistic Regression</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        # Sigmoid curve
        z_vals = np.linspace(-8, 8, 500)
        sig_vals = sigmoid(z_vals)

        fig_sig = go.Figure()
        # Gradient fill
        fig_sig.add_trace(go.Scatter(
            x=z_vals, y=sig_vals,
            mode="lines",
            line=dict(color="#63b3ed", width=3),
            fill="tozeroy",
            fillcolor="rgba(99,179,237,0.08)",
            name="σ(z)",
            hovertemplate="z = %{x:.2f}<br>σ(z) = %{y:.4f}<extra></extra>"
        ))
        # Decision boundary
        fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#f6e05e",
                          annotation_text="Decision Boundary (0.5)", annotation_font_color="#f6e05e")
        fig_sig.add_vline(x=0, line_dash="dot", line_color="rgba(226,232,240,0.25)")

        fig_sig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="σ(z) = 1 / (1 + e⁻ᶻ)", font=dict(size=16, color="#b794f4")),
            xaxis_title="z  (Linear Combination of Features)",
            yaxis_title="Probability of Default",
            height=360,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_r:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:1.05rem;font-weight:700;color:#b794f4;margin-bottom:14px;'>📐 Sigmoid Formula</div>
            <div class='formula-box'>σ(z) = 1 / (1 + e⁻ᶻ)</div>
            <div style='margin-top:14px;font-size:0.9rem;line-height:1.85;color:rgba(226,232,240,0.8);'>
                <b style='color:#63b3ed;'>z</b> = Linear combination of features<br>
                <b style='color:#63b3ed;'>Output</b>: Always between 0 and 1<br><br>
                <b>Where z =</b><br>
                β₀ + β₁·CreditScore + β₂·Income<br>
                &nbsp;&nbsp;&nbsp;+ β₃·DTI + β₄·Employment + ...
            </div>
        </div>
        <div class='glass-card'>
            <div style='font-size:1.05rem;font-weight:700;color:#68d391;margin-bottom:10px;'>⚙ Pre-computed Coefficients</div>
            <table style='width:100%;font-size:0.82rem;'>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₀ (Intercept)</td><td style='color:#f6e05e;text-align:right;font-family:monospace;'>+3.500</td></tr>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₁ Credit Score</td><td style='color:#fc8181;text-align:right;font-family:monospace;'>−0.008</td></tr>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₂ Income</td><td style='color:#fc8181;text-align:right;font-family:monospace;'>−0.000012</td></tr>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₃ Debt-to-Income</td><td style='color:#f6e05e;text-align:right;font-family:monospace;'>+3.200</td></tr>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₄ Employment Yrs</td><td style='color:#fc8181;text-align:right;font-family:monospace;'>−0.060</td></tr>
                <tr><td style='color:rgba(226,232,240,0.6);padding:4px 0;'>β₅ Interest Rate</td><td style='color:#f6e05e;text-align:right;font-family:monospace;'>+0.040</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # PD Distribution
    st.markdown("<div class='section-header'>Probability of Default — Distribution Across Portfolio</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pd_hist = go.Figure()
        fig_pd_hist.add_trace(go.Histogram(
            x=df[df["default"] == 0]["prob_default"],
            nbinsx=40, name="No Default",
            marker_color="rgba(99,179,237,0.7)", opacity=0.8
        ))
        fig_pd_hist.add_trace(go.Histogram(
            x=df[df["default"] == 1]["prob_default"],
            nbinsx=40, name="Default",
            marker_color="rgba(252,129,129,0.7)", opacity=0.8
        ))
        fig_pd_hist.update_layout(
            **PLOTLY_LAYOUT, barmode="overlay",
            title=dict(text="PD Distribution — Defaults vs Non-Defaults", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="P(Default)", yaxis_title="Count", height=320,
        )
        st.plotly_chart(fig_pd_hist, use_container_width=True)

    with col_b:
        # Credit Score vs PD scatter
        sample = df.sample(300, random_state=1)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=sample["credit_score"], y=sample["prob_default"],
            mode="markers",
            marker=dict(
                color=sample["prob_default"], colorscale="RdYlGn_r",
                size=6, opacity=0.75,
                colorbar=dict(title="PD", tickfont=dict(color="#e2e8f0"))
            ),
            hovertemplate="Credit: %{x:.0f}<br>PD: %{y:.3f}<extra></extra>",
            name="Customers"
        ))
        # Trend line (analytical)
        cs = np.linspace(300, 850, 200)
        pd_trend = sigmoid(3.5 - 0.008 * cs)
        fig_scatter.add_trace(go.Scatter(
            x=cs, y=pd_trend, mode="lines",
            line=dict(color="#f6e05e", width=2, dash="dash"),
            name="Logistic Trend"
        ))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Credit Score vs Probability of Default", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="Credit Score", yaxis_title="P(Default)", height=320,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Normal Distribution Section
    st.markdown("<div class='section-header'>Normal Distribution — Credit Score & Income</div>", unsafe_allow_html=True)
    col_n1, col_n2 = st.columns(2)

    with col_n1:
        x_cs = np.linspace(300, 850, 400)
        mu_cs, sig_cs = df["credit_score"].mean(), df["credit_score"].std()
        pdf_cs = norm.pdf(x_cs, mu_cs, sig_cs)

        fig_norm_cs = go.Figure()
        fig_norm_cs.add_trace(go.Histogram(
            x=df["credit_score"], histnorm="probability density",
            nbinsx=35, marker_color="rgba(99,179,237,0.45)",
            name="Actual", opacity=0.8
        ))
        fig_norm_cs.add_trace(go.Scatter(
            x=x_cs, y=pdf_cs, mode="lines",
            line=dict(color="#b794f4", width=2.5), name=f"N({mu_cs:.0f}, {sig_cs:.0f}²)"
        ))
        fig_norm_cs.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Credit Score — Normal Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="Credit Score", yaxis_title="Probability Density", height=300,
        )
        st.plotly_chart(fig_norm_cs, use_container_width=True)

    with col_n2:
        x_inc = np.linspace(0, 160000, 400)
        mu_inc, sig_inc = df["income"].mean(), df["income"].std()
        pdf_inc = norm.pdf(x_inc, mu_inc, sig_inc)

        fig_norm_inc = go.Figure()
        fig_norm_inc.add_trace(go.Histogram(
            x=df["income"], histnorm="probability density",
            nbinsx=35, marker_color="rgba(104,211,145,0.45)",
            name="Actual", opacity=0.8
        ))
        fig_norm_inc.add_trace(go.Scatter(
            x=x_inc, y=pdf_inc, mode="lines",
            line=dict(color="#f687b3", width=2.5), name=f"N({mu_inc/1000:.0f}k, {sig_inc/1000:.0f}k²)"
        ))
        fig_norm_inc.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Annual Income — Normal Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="Income (₹)", yaxis_title="Probability Density", height=300,
        )
        st.plotly_chart(fig_norm_inc, use_container_width=True)


# ═══════════════════════════════════════════════
#  TAB 2 — PORTFOLIO ANALYTICS
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Portfolio Risk Overview</div>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # Default vs Non-Default Pie
        default_counts = df["default"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["No Default", "Default"],
            values=[default_counts.get(0, 0), default_counts.get(1, 0)],
            marker_colors=["#63b3ed", "#fc8181"],
            hole=0.55,
            textfont=dict(size=13, color="#e2e8f0"),
            hovertemplate="%{label}: %{value} loans (%{percent})<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"{df['default'].mean()*100:.1f}%<br>Default Rate",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#fc8181", family="Inter")
        )
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Portfolio Default Distribution", font=dict(size=14, color="#e2e8f0")),
            height=340, showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_p2:
        # Risk Bucketing
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels_risk = ["Very Low\n<20%", "Low\n20-40%", "Medium\n40-60%", "High\n60-80%", "Very High\n>80%"]
        df["risk_bucket"] = pd.cut(df["prob_default"], bins=bins, labels=labels_risk)
        bucket_counts = df["risk_bucket"].value_counts().sort_index()
        colors_risk = ["#48bb78", "#68d391", "#f6e05e", "#f6ad55", "#fc8181"]

        fig_bucket = go.Figure(go.Bar(
            x=bucket_counts.index.astype(str),
            y=bucket_counts.values,
            marker_color=colors_risk,
            text=bucket_counts.values,
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
            hovertemplate="%{x}: %{y} loans<extra></extra>",
        ))
        fig_bucket.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Risk Bucket Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="Risk Category", yaxis_title="Number of Loans", height=340,
        )
        st.plotly_chart(fig_bucket, use_container_width=True)

    col_p3, col_p4 = st.columns(2)

    with col_p3:
        # Loan Amount vs Default
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[df["default"] == 0]["loan_amount"],
            name="No Default", marker_color="#63b3ed",
            boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=df[df["default"] == 1]["loan_amount"],
            name="Default", marker_color="#fc8181",
            boxmean=True,
        ))
        fig_box.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Loan Amount Distribution by Default Status", font=dict(size=14, color="#e2e8f0")),
            yaxis_title="Loan Amount (₹)", height=330,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_p4:
        # Correlation Heatmap
        corr_cols = ["credit_score", "income", "loan_amount", "debt_to_income",
                     "employment_years", "interest_rate", "prob_default"]
        corr_matrix = df[corr_cols].corr().round(2)
        short_labels = ["Credit", "Income", "Loan Amt", "DTI", "Employ", "Int Rate", "PD"]

        fig_heat = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=short_labels, y=short_labels,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#e2e8f0"),
            hovertemplate="r = %{z:.2f}<extra></extra>",
            colorbar=dict(title="r", tickfont=dict(color="#e2e8f0")),
        ))
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Correlation Matrix — r = Cov(X,Y) / (σx·σy)", font=dict(size=13, color="#e2e8f0")),
            height=330,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # DTI vs Income scatter coloured by default
    st.markdown("<div class='section-header'>Income vs Debt-to-Income Ratio — Risk Map</div>", unsafe_allow_html=True)
    sample2 = df.sample(400, random_state=7)
    fig_scatter2 = go.Figure()
    for label, color, marker in [(0, "#63b3ed", "circle"), (1, "#fc8181", "x")]:
        sub = sample2[sample2["default"] == label]
        fig_scatter2.add_trace(go.Scatter(
            x=sub["income"], y=sub["debt_to_income"],
            mode="markers",
            marker=dict(color=color, size=7, symbol=marker, opacity=0.7),
            name=["No Default", "Default"][label],
            hovertemplate="Income: ₹%{x:,.0f}<br>DTI: %{y:.2f}<extra></extra>",
        ))
    fig_scatter2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Income vs DTI — Default Risk Scatter", font=dict(size=14, color="#e2e8f0")),
        xaxis_title="Annual Income (₹)", yaxis_title="Debt-to-Income Ratio", height=360,
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)


# ═══════════════════════════════════════════════
#  TAB 3 — EXPECTED LOSS MODEL
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Expected Loss = PD × LGD × EAD</div>", unsafe_allow_html=True)

    # Formula Banner
    st.markdown("""
    <div class='glass-card' style='text-align:center;'>
        <div class='formula-box' style='font-size:1.6rem; padding:24px;'>EL = PD × LGD × EAD</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:20px;'>
            <div>
                <div style='font-size:2rem;font-weight:800;color:#63b3ed;'>PD</div>
                <div style='color:rgba(226,232,240,0.7);font-size:0.88rem;'>Probability of Default<br>Estimated via Sigmoid</div>
            </div>
            <div>
                <div style='font-size:2rem;font-weight:800;color:#f6e05e;'>LGD</div>
                <div style='color:rgba(226,232,240,0.7);font-size:0.88rem;'>Loss Given Default<br>= 1 − Recovery Rate</div>
            </div>
            <div>
                <div style='font-size:2rem;font-weight:800;color:#f687b3;'>EAD</div>
                <div style='color:rgba(226,232,240,0.7);font-size:0.88rem;'>Exposure at Default<br>Outstanding Balance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_el1, col_el2 = st.columns(2)

    with col_el1:
        # EL distribution
        fig_el_hist = go.Figure()
        fig_el_hist.add_trace(go.Histogram(
            x=df["expected_loss"], nbinsx=50,
            marker=dict(
                color=df.sort_values("expected_loss")["prob_default"],
                colorscale="YlOrRd",
                line=dict(width=0),
            ),
            opacity=0.85, name="Expected Loss",
            hovertemplate="EL: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        fig_el_hist.add_vline(
            x=df["expected_loss"].mean(),
            line_dash="dash", line_color="#f6e05e",
            annotation_text=f"Mean EL: ₹{df['expected_loss'].mean():,.0f}",
            annotation_font_color="#f6e05e",
        )
        fig_el_hist.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Expected Loss Distribution", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="Expected Loss (₹)", yaxis_title="Count", height=330,
        )
        st.plotly_chart(fig_el_hist, use_container_width=True)

    with col_el2:
        # PD vs EAD Bubble — size = EL
        sample3 = df.sample(200, random_state=5)
        fig_bubble = go.Figure(go.Scatter(
            x=sample3["ead"], y=sample3["prob_default"],
            mode="markers",
            marker=dict(
                size=np.clip(sample3["expected_loss"] / 3000, 3, 25),
                color=sample3["expected_loss"],
                colorscale="YlOrRd",
                opacity=0.7,
                colorbar=dict(title="EL (₹)", tickfont=dict(color="#e2e8f0")),
            ),
            hovertemplate="EAD: ₹%{x:,.0f}<br>PD: %{y:.3f}<br>EL: ₹%{marker.color:,.0f}<extra></extra>",
        ))
        fig_bubble.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="PD vs EAD — Bubble Size = Expected Loss", font=dict(size=13, color="#e2e8f0")),
            xaxis_title="EAD (₹)", yaxis_title="PD", height=330,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Portfolio EL by Risk Bucket
    el_by_bucket = df.groupby("risk_bucket", observed=True).agg(
        total_el=("expected_loss", "sum"),
        count=("expected_loss", "count"),
        avg_pd=("prob_default", "mean"),
    ).reset_index()

    col_el3, col_el4 = st.columns(2)
    with col_el3:
        fig_el_bucket = go.Figure(go.Bar(
            x=el_by_bucket["risk_bucket"].astype(str),
            y=el_by_bucket["total_el"],
            marker_color=["#48bb78", "#68d391", "#f6e05e", "#f6ad55", "#fc8181"][:len(el_by_bucket)],
            text=[f"₹{v/1000:.0f}K" for v in el_by_bucket["total_el"]],
            textposition="outside", textfont=dict(color="#e2e8f0"),
            hovertemplate="%{x}<br>Total EL: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_el_bucket.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Portfolio Expected Loss by Risk Bucket", font=dict(size=13, color="#e2e8f0")),
            xaxis_title="Risk Bucket", yaxis_title="Total Expected Loss (₹)", height=320,
        )
        st.plotly_chart(fig_el_bucket, use_container_width=True)

    with col_el4:
        # LGD distribution
        fig_lgd = go.Figure()
        x_lgd = np.linspace(0, 1, 300)
        # Beta-shaped LGD density line
        from scipy.stats import beta as beta_dist
        lgd_pdf = beta_dist.pdf(x_lgd, 2, 3)
        fig_lgd.add_trace(go.Histogram(
            x=df["lgd"], histnorm="probability density",
            nbinsx=30, marker_color="rgba(246,224,94,0.45)", name="LGD", opacity=0.85
        ))
        fig_lgd.add_trace(go.Scatter(
            x=x_lgd, y=lgd_pdf, mode="lines",
            line=dict(color="#f6e05e", width=2.5), name="Beta(2,3) fit"
        ))
        fig_lgd.add_vline(
            x=df["lgd"].mean(),
            line_dash="dash", line_color="#fc8181",
            annotation_text=f"Mean LGD = {df['lgd'].mean():.2f}",
            annotation_font_color="#fc8181",
        )
        fig_lgd.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Loss Given Default (LGD) Distribution", font=dict(size=13, color="#e2e8f0")),
            xaxis_title="LGD", yaxis_title="Density", height=320,
        )
        st.plotly_chart(fig_lgd, use_container_width=True)

    # Summary stats table
    st.markdown("<div class='section-header'>Portfolio Statistics — Mean, Variance, Std Dev</div>", unsafe_allow_html=True)
    stats_data = []
    for col_name, label, prefix in [
        ("prob_default", "Probability of Default", ""),
        ("expected_loss", "Expected Loss", "₹"),
        ("loan_amount", "Loan Amount", "₹"),
        ("credit_score", "Credit Score", ""),
        ("income", "Annual Income", "₹"),
        ("lgd", "Loss Given Default", ""),
        ("ead", "Exposure at Default", "₹"),
    ]:
        vals = df[col_name]
        mean = vals.mean()
        var  = vals.var()
        std  = vals.std()
        stats_data.append({
            "Feature": label,
            "Mean (x̄)":         f"{prefix}{mean:,.2f}",
            "Variance (σ²)":    f"{prefix}{var:,.2f}",
            "Std Dev (σ)":      f"{prefix}{std:,.2f}",
            "Min":              f"{prefix}{vals.min():,.2f}",
            "Max":              f"{prefix}{vals.max():,.2f}",
        })
    st.dataframe(
        pd.DataFrame(stats_data),
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════
#  TAB 4 — LOAN PREDICTOR
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🎯 Individual Loan Default Predictor</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card' style='margin-bottom:10px;'>
        <span style='color:rgba(226,232,240,0.65);font-size:0.9rem;'>
        Enter customer details below. The model applies <b style='color:#63b3ed;'>pre-computed logistic regression coefficients</b>
        through the <b style='color:#b794f4;'>sigmoid function</b> to instantly compute the Probability of Default and Expected Loss.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        credit_score_in = st.slider("Credit Score",       300, 850, 650, 5)
        income_in       = st.number_input("Annual Income (₹)", 10000, 200000, 60000, 5000)
        loan_amount_in  = st.number_input("Loan Amount (₹)",   10000, 1000000, 150000, 10000)

    with col_f2:
        dti_in          = st.slider("Debt-to-Income Ratio", 0.05, 0.95, 0.35, 0.01)
        employment_in   = st.slider("Employment Years",      0, 40, 5, 1)
        interest_in     = st.slider("Interest Rate (%)",     4.0, 24.0, 10.5, 0.5)

    with col_f3:
        num_credit_in   = st.slider("Number of Credit Lines", 1, 12, 4, 1)
        lgd_in          = st.slider("Loss Given Default",     0.10, 0.90, 0.45, 0.05)
        recovery_rate   = 1 - lgd_in
        ead_in          = loan_amount_in * 0.90      # 90% outstanding

    # Compute
    pd_val, z_val = compute_pd(
        credit_score_in, income_in, dti_in,
        employment_in, interest_in, num_credit_in
    )
    el_val = pd_val * lgd_in * ead_in

    # Risk classification
    if pd_val < 0.25:
        risk_label, risk_class = "🟢 LOW RISK", "risk-low"
    elif pd_val < 0.55:
        risk_label, risk_class = "🟡 MEDIUM RISK", "risk-medium"
    else:
        risk_label, risk_class = "🔴 HIGH RISK", "risk-high"

    st.markdown("<br>", unsafe_allow_html=True)
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value accent-{"red" if pd_val>0.5 else "green"}'>{pd_val*100:.1f}%</div>
            <div class='kpi-label'>Probability of Default</div>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value accent-yellow'>₹{el_val:,.0f}</div>
            <div class='kpi-label'>Expected Loss (EL)</div>
        </div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value accent-purple'>{z_val:+.3f}</div>
            <div class='kpi-label'>Logit Score (z)</div>
        </div>""", unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value accent-green'>{recovery_rate*100:.0f}%</div>
            <div class='kpi-label'>Recovery Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='glass-card' style='text-align:center; margin-top:16px;'>
        <div style='font-size:1.05rem; color:rgba(226,232,240,0.6); margin-bottom:10px;'>Decision</div>
        <span class='risk-badge {risk_class}' style='font-size:1.2rem; padding:12px 36px;'>{risk_label}</span>
        <div style='margin-top:18px; font-size:0.9rem; color:rgba(226,232,240,0.6);'>
            σ(z) = 1 / (1 + e<sup>−{z_val:+.3f}</sup>) = <b style='color:#b794f4;'>{pd_val:.4f}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            EL = {pd_val:.3f} × {lgd_in:.2f} × ₹{ead_in:,.0f} = <b style='color:#f6e05e;'>₹{el_val:,.0f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sigmoid gauge chart
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pd_val * 100,
            number=dict(suffix="%", font=dict(size=40, color="#e2e8f0")),
            delta=dict(reference=50, valueformat=".1f", suffix="%"),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#a0aec0", tickfont=dict(color="#a0aec0")),
                bar=dict(color="#fc8181" if pd_val > 0.5 else "#63b3ed", thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(99,179,237,0.2)",
                steps=[
                    dict(range=[0, 25],  color="rgba(72,187,120,0.15)"),
                    dict(range=[25, 55], color="rgba(246,224,94,0.12)"),
                    dict(range=[55, 100],color="rgba(252,129,129,0.15)"),
                ],
                threshold=dict(line=dict(color="#f6e05e", width=3), thickness=0.8, value=50),
            ),
            title=dict(text="Probability of Default (%)", font=dict(color="#e2e8f0", size=14)),
        ))
        fig_gauge.update_layout(
            **PLOTLY_LAYOUT,
            height=300, margin=dict(l=30, r=30, t=50, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        # Show sigmoid curve with current customer marked
        z_range = np.linspace(-8, 8, 400)
        sig_range = sigmoid(z_range)
        fig_sig2 = go.Figure()
        fig_sig2.add_trace(go.Scatter(
            x=z_range, y=sig_range, mode="lines",
            line=dict(color="#63b3ed", width=2.5),
            name="σ(z)", fill="tozeroy", fillcolor="rgba(99,179,237,0.06)",
        ))
        fig_sig2.add_trace(go.Scatter(
            x=[z_val], y=[pd_val],
            mode="markers+text",
            marker=dict(color="#fc8181" if pd_val > 0.5 else "#68d391", size=14,
                        symbol="star", line=dict(color="#fff", width=1.5)),
            text=[f"  z={z_val:.2f}<br>σ={pd_val:.3f}"],
            textfont=dict(color="#e2e8f0", size=11),
            textposition="middle right",
            name="This Customer",
        ))
        fig_sig2.add_hline(y=0.5, line_dash="dash", line_color="#f6e05e",
                           annotation_text="Threshold (0.5)", annotation_font_color="#f6e05e")
        fig_sig2.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Customer Position on Sigmoid Curve", font=dict(size=13, color="#e2e8f0")),
            xaxis_title="z", yaxis_title="P(Default)", height=300,
        )
        st.plotly_chart(fig_sig2, use_container_width=True)


# ═══════════════════════════════════════════════
#  TAB 5 — MODEL METRICS
# ═══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Model Evaluation Metrics</div>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1.3, 1])

    with col_m1:
        # Confusion Matrix Heatmap
        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: No Default", "Predicted: Default"],
            y=["Actual: No Default", "Actual: Default"],
            colorscale=[[0,"rgba(99,179,237,0.08)"], [1,"rgba(99,179,237,0.55)"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=22, color="#e2e8f0"),
            hovertemplate="%{y} / %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Confusion Matrix", font=dict(size=15, color="#e2e8f0")),
            height=340,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        st.markdown(f"""
        <div class='glass-card'>
            <div style='font-size:1.05rem;font-weight:700;color:#63b3ed;margin-bottom:18px;'>📊 Performance Metrics</div>
            <table style='width:100%;font-size:0.95rem;line-height:2.2;'>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>✅ Accuracy</td>
                    <td style='color:#68d391;font-weight:700;text-align:right;font-size:1.1rem;'>{metrics["acc"]*100:.2f}%</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>🎯 Precision</td>
                    <td style='color:#63b3ed;font-weight:700;text-align:right;font-size:1.1rem;'>{metrics["prec"]*100:.2f}%</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>🔍 Recall</td>
                    <td style='color:#b794f4;font-weight:700;text-align:right;font-size:1.1rem;'>{metrics["rec"]*100:.2f}%</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>⚖ F1 Score</td>
                    <td style='color:#f6e05e;font-weight:700;text-align:right;font-size:1.1rem;'>{metrics["f1"]*100:.2f}%</td>
                </tr>
                <tr><td colspan='2'><hr style='border-color:rgba(99,179,237,0.15);margin:8px 0;'/></td></tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>True Positives</td>
                    <td style='color:#fc8181;font-weight:700;text-align:right;'>{metrics["tp"]}</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>True Negatives</td>
                    <td style='color:#68d391;font-weight:700;text-align:right;'>{metrics["tn"]}</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>False Positives</td>
                    <td style='color:#f6e05e;font-weight:700;text-align:right;'>{metrics["fp"]}</td>
                </tr>
                <tr>
                    <td style='color:rgba(226,232,240,0.65);'>False Negatives</td>
                    <td style='color:#f6ad55;font-weight:700;text-align:right;'>{metrics["fn"]}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ROC-like curve (analytical using sigmoid scores)
    st.markdown("<div class='section-header'>ROC Curve (Analytical)</div>", unsafe_allow_html=True)

    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for t in thresholds:
        pred_t = (df["prob_default"] >= t).astype(int)
        tp_t = int(((pred_t == 1) & (df["default"] == 1)).sum())
        fn_t = int(((pred_t == 0) & (df["default"] == 1)).sum())
        fp_t = int(((pred_t == 1) & (df["default"] == 0)).sum())
        tn_t = int(((pred_t == 0) & (df["default"] == 0)).sum())
        tprs.append(tp_t / (tp_t + fn_t + 1e-9))
        fprs.append(fp_t / (fp_t + tn_t + 1e-9))

    roc_auc = float(np.trapz(tprs[::-1], fprs[::-1]))

    col_roc1, col_roc2 = st.columns(2)
    with col_roc1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fprs, y=tprs,
            mode="lines", name=f"ROC (AUC={roc_auc:.3f})",
            line=dict(color="#b794f4", width=2.5),
            fill="tozeroy", fillcolor="rgba(183,148,244,0.08)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="rgba(226,232,240,0.25)", dash="dash"),
            name="Random Classifier",
        ))
        fig_roc.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"ROC Curve — AUC = {roc_auc:.3f}", font=dict(size=14, color="#e2e8f0")),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=340,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_roc2:
        # Precision-Recall trade-off across thresholds
        precs, recs = [], []
        for t in thresholds:
            pred_t = (df["prob_default"] >= t).astype(int)
            tp_t = int(((pred_t == 1) & (df["default"] == 1)).sum())
            fp_t = int(((pred_t == 1) & (df["default"] == 0)).sum())
            fn_t = int(((pred_t == 0) & (df["default"] == 1)).sum())
            precs.append(tp_t / (tp_t + fp_t + 1e-9))
            recs.append(tp_t / (tp_t + fn_t + 1e-9))

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=thresholds, y=precs,
            mode="lines", name="Precision",
            line=dict(color="#63b3ed", width=2),
        ))
        fig_pr.add_trace(go.Scatter(
            x=thresholds, y=recs,
            mode="lines", name="Recall",
            line=dict(color="#f687b3", width=2),
        ))
        fig_pr.add_vline(x=0.5, line_dash="dash", line_color="#f6e05e",
                         annotation_text="Current Threshold", annotation_font_color="#f6e05e")
        fig_pr.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Precision & Recall vs Threshold", font=dict(size=13, color="#e2e8f0")),
            xaxis_title="Decision Threshold", yaxis_title="Score", height=340,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Feature importance (coefficient magnitudes)
    st.markdown("<div class='section-header'>Feature Importance — Coefficient Magnitudes |β|</div>", unsafe_allow_html=True)
    features     = ["Credit Score", "Debt-to-Income", "Employment Yrs", "Interest Rate", "Income", "Num Credit Lines"]
    coefficients = [-0.008, 3.2, -0.06, 0.04, -0.000012 * 10000, -0.002]
    importance   = np.abs(coefficients)
    sort_idx     = np.argsort(importance)[::-1]

    fig_feat = go.Figure(go.Bar(
        y=[features[i] for i in sort_idx],
        x=[importance[i] for i in sort_idx],
        orientation="h",
        marker=dict(
            color=[importance[i] for i in sort_idx],
            colorscale="Blues",
            line=dict(width=0),
        ),
        text=[f"β = {coefficients[i]:+.4f}" for i in sort_idx],
        textposition="outside", textfont=dict(color="#e2e8f0"),
        hovertemplate="%{y}: |β| = %{x:.4f}<extra></extra>",
    ))
    fig_feat.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Feature Importance via Logistic Regression Coefficients", font=dict(size=13, color="#e2e8f0")),
        xaxis_title="|Coefficient Value|", height=320,
    )
    st.plotly_chart(fig_feat, use_container_width=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:30px 0 10px; color:rgba(226,232,240,0.3); font-size:0.8rem;'>
    Probabilistic Default Modelling for Loan Portfolios &nbsp;|&nbsp;
    Statistics Project &nbsp;|&nbsp;
    Logistic Regression · Sigmoid · EL = PD × LGD × EAD · Normal Distribution · Correlation
</div>
""", unsafe_allow_html=True)
