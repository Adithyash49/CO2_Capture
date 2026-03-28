# === CO2_Capture_Hybrid.py ===
# Enhanced interactive dashboard with insights, recommendations, model comparison

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pathlib
from scipy.optimize import minimize

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='CO₂ Capture Dashboard',
    page_icon='🏭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0F1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E2130, #252836);
        border: 1px solid #2E3250;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Insight boxes */
    .insight-green {
        background: linear-gradient(135deg, #0D2B1F, #0F3D28);
        border-left: 4px solid #00C875;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .insight-orange {
        background: linear-gradient(135deg, #2B1A0D, #3D2810);
        border-left: 4px solid #FF8C00;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .insight-red {
        background: linear-gradient(135deg, #2B0D0D, #3D1010);
        border-left: 4px solid #FF4444;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .insight-blue {
        background: linear-gradient(135deg, #0D1A2B, #102030);
        border-left: 4px solid #4A90D9;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8892A4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent

@st.cache_resource
def load_models():
    return {
        'capture':    joblib.load(SCRIPT_DIR / 'capture_surrogate.pkl'),
        'qreb':       joblib.load(SCRIPT_DIR / 'qreb_surrogate.pkl'),
        'correction': joblib.load(SCRIPT_DIR / 'correction_A.pkl'),
        'param_C':    joblib.load(SCRIPT_DIR / 'param_model_C.pkl'),
    }

models = load_models()

EF = {'Germany':0.40, 'France':0.05, 'Norway':0.02, 'Poland':0.78, 'India':0.71}

# ── Physics helpers ───────────────────────────────────────────────────────────
def physics_qreb(amine, lg, t_abs, p_str):
    Q = (-2.402 + 0.849/lg + 0.039*lg**2 + 0.120*t_abs + 0.723/p_str)
    return float(np.clip(Q, 2.5, 8.0))

def predict_all(amine, lg, t_abs, p_str, flue):
    X = np.array([[amine, lg, t_abs, p_str, flue]])
    cap    = float(models['capture'].predict(X)[0])
    qreb_ml= float(models['qreb'].predict(X)[0])
    qreb_ph= physics_qreb(amine, lg, t_abs, p_str)
    corr   = float(models['correction'].predict(X)[0])
    qreb_A = qreb_ph + corr
    da_ml  = float(np.clip(models['param_C'].predict(X)[0], 0.05, 0.50))
    Qs_C   = lg*3.5*(120-t_abs)*44.01/(da_ml*1000)/1000
    qreb_C = float(np.clip(Qs_C + 85*1000/44.01/1e6 + 0.8/p_str, 2.5, 8.0))
    return cap, qreb_ph, qreb_ml, qreb_A, qreb_C, da_ml

def calc_lca(qreb, lg, t_abs, p_str, amine, ef):
    elec       = (88 + 12*(p_str-1.5)) + (12 + 5*lg)
    sloss      = max(0.3, 0.3 + 0.022*t_abs + 0.016*amine)
    gwp_steam  = qreb * 1000 * 56.1 / 1000
    gwp_elec   = elec * ef
    gwp_mea    = sloss * 2.49
    gwp_const  = 15.0
    gwp_total  = gwp_steam + gwp_elec + gwp_mea + gwp_const
    net        = 1000 - gwp_total
    return gwp_total, net, gwp_steam, gwp_elec, gwp_mea, gwp_const

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏭 CO₂ Capture — Hybrid Model Dashboard")
st.markdown("MEA Post-Combustion Capture &nbsp;·&nbsp; 300-run parametric dataset &nbsp;·&nbsp; TCM Mongstad · DTU WtE · Aker MTU")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Process Inputs")
    st.markdown('<p class="section-header">Operating Conditions</p>', unsafe_allow_html=True)

    amine = st.slider('MEA Concentration (wt%)',  20.0, 40.0, 30.0, 0.5,
                      help='Weight % of MEA in water-solvent mixture. Industry standard: 30 wt%')
    lg    = st.slider('L/G Ratio (mol/mol)',       1.0,  5.5,  3.0, 0.1,
                      help='Moles of liquid solvent per mole of flue gas. Sweet spot: 2.5–3.5')
    t_abs = st.slider('Absorber Temperature (°C)', 35.0, 60.0, 45.0, 0.5,
                      help='Lower = better CO₂ solubility. Optimum: 40–45°C')
    p_str = st.slider('Stripper Pressure (bar)',   1.3,  2.4,  1.8, 0.05,
                      help='Higher pressure = less steam needed for regeneration')
    flue  = st.slider('Flue Gas CO₂ (%)',          7.0,  13.0, 10.0, 0.5,
                      help='CO₂ vol% in flue gas. WtE plants: 8–12%')

    st.divider()
    st.markdown('<p class="section-header">LCA Settings</p>', unsafe_allow_html=True)
    country = st.selectbox('Electricity Grid', list(EF.keys()), index=0)
    ef      = EF[country]

    st.divider()
    st.caption("📌 Dataset: CO2_Capture_Dataset_v3.xlsx")
    st.caption("📌 300 Aspen Plus parametric runs")
    st.caption("📌 Physics R²=0.82 → Hybrid R²=0.97")

# ── Compute predictions ───────────────────────────────────────────────────────
cap, qreb_ph, qreb_ml, qreb_A, qreb_C, da_ml = predict_all(amine, lg, t_abs, p_str, flue)
gwp_tot, net, gwp_steam, gwp_elec, gwp_mea, gwp_const = calc_lca(
    qreb_A, lg, t_abs, p_str, amine, ef)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🎯 Predictions & Models',
    '🌿 LCA & Environment',
    '💡 Process Insights',
    '⚡ Optimisation'
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════
with tab1:

    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Capture Rate",  f"{cap:.1f}%",
                delta=f"{cap-85:.1f}% vs target",
                delta_color="normal")
    col2.metric("Physics",       f"{qreb_ph:.3f} GJ/t",  help="Simplified Oexmann model (R²=0.82)")
    col3.metric("Pure ML",       f"{qreb_ml:.3f} GJ/t",  help="XGBoost surrogate (R²=0.95)")
    col4.metric("Hybrid A",      f"{qreb_A:.3f} GJ/t",   help="Physics + ML correction (R²=0.97)")
    col5.metric("Hybrid C",      f"{qreb_C:.3f} GJ/t",   help="ML-predicted delta_alpha (R²=0.97)")

    st.divider()
    left, right = st.columns([1, 1])

    # Left: model comparison bar chart
    with left:
        st.markdown("#### Model Comparison")
        models_names = ['Physics\n(R²=0.82)', 'Pure ML\n(R²=0.95)',
                        'Hybrid A\n(R²=0.97)', 'Hybrid C\n(R²=0.97)']
        values = [qreb_ph, qreb_ml, qreb_A, qreb_C]
        colours = ['#6B7280', '#4A90D9', '#F97316', '#10B981']

        fig_bar = go.Figure()
        for i, (name, val, col) in enumerate(zip(models_names, values, colours)):
            fig_bar.add_bar(
                x=[name], y=[val], marker_color=col,
                text=[f'{val:.3f}'], textposition='outside',
                name=name
            )
        fig_bar.add_hline(y=4.0, line_dash='dash', line_color='red',
                         annotation_text='Industry target 4.0 GJ/t',
                         annotation_position='top right')
        fig_bar.update_layout(
            title='Q_reboiler predicted by all 4 models',
            yaxis_title='Q_reboiler (GJ/t CO₂)',
            yaxis_range=[0, max(values) * 1.3],
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=380
        )
        fig_bar.update_xaxes(gridcolor='#2E3250')
        fig_bar.update_yaxes(gridcolor='#2E3250')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Right: U-shape sweep
    with right:
        st.markdown("#### U-Shape: Q_reboiler vs L/G (at your current conditions)")
        sweep_lg  = np.linspace(1.0, 5.5, 60)
        ph_sweep  = [physics_qreb(amine, l, t_abs, p_str) for l in sweep_lg]
        ml_sweep  = [float(models['qreb'].predict(
                     np.array([[amine,l,t_abs,p_str,flue]]))[0]) for l in sweep_lg]
        A_sweep   = [physics_qreb(amine,l,t_abs,p_str) +
                     float(models['correction'].predict(
                     np.array([[amine,l,t_abs,p_str,flue]]))[0]) for l in sweep_lg]

        fig_u = go.Figure()
        fig_u.add_scatter(x=sweep_lg, y=ph_sweep, name='Physics',
                          line=dict(color='#6B7280', width=2, dash='dot'))
        fig_u.add_scatter(x=sweep_lg, y=ml_sweep,  name='Pure ML',
                          line=dict(color='#4A90D9', width=2))
        fig_u.add_scatter(x=sweep_lg, y=A_sweep,   name='Hybrid A',
                          line=dict(color='#F97316', width=2.5))
        fig_u.add_vline(x=lg, line_dash='dash', line_color='white',
                        annotation_text=f'Current L/G={lg:.1f}',
                        annotation_font_color='white')
        fig_u.add_hline(y=4.0, line_dash='dash', line_color='red', opacity=0.5)
        fig_u.update_layout(
            title='U-shape: sweet spot L/G = 2.5–3.5',
            xaxis_title='L/G Ratio (mol/mol)',
            yaxis_title='Q_reboiler (GJ/t CO₂)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=380,
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        fig_u.update_xaxes(gridcolor='#2E3250')
        fig_u.update_yaxes(gridcolor='#2E3250')
        st.plotly_chart(fig_u, use_container_width=True)

    # Architecture C insight
    st.divider()
    st.markdown("#### 🔍 What Architecture C Reveals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Physics assumed delta_α", "0.200 (fixed)")
    c2.metric("ML predicted delta_α",    f"{da_ml:.3f}")
    c3.metric("Difference",              f"{da_ml - 0.20:+.3f}",
              delta_color="off")

    if da_ml > 0.22:
        st.markdown(f"""<div class="insight-green">
        ✅ <b>Better than assumed:</b> Real working capacity ({da_ml:.3f}) is higher than
        physics assumed (0.200). Your solvent is performing well — absorbing more CO₂ per
        mole of MEA than the simple model expected. Physics would overestimate Q_reboiler here.
        </div>""", unsafe_allow_html=True)
    elif da_ml < 0.18:
        st.markdown(f"""<div class="insight-red">
        ⚠️ <b>Worse than assumed:</b> Real working capacity ({da_ml:.3f}) is lower than
        physics assumed (0.200). Possible causes: high absorber temperature degrading MEA,
        solvent approaching loading limit, or poor L/G choice. Physics would underestimate
        Q_reboiler here — real energy cost is higher.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="insight-blue">
        ℹ️ <b>As expected:</b> Working capacity ({da_ml:.3f}) is close to the 0.200 assumption.
        Physics and data agree well at these conditions.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — LCA
# ════════════════════════════════════════════════════════════════════
with tab2:

    col1, col2, col3 = st.columns(3)
    col1.metric("GWP Total",     f"{gwp_tot:.0f} kg CO₂-eq/t")
    col2.metric("Net Avoided",   f"{net:.0f} kg CO₂-eq/t")
    col3.metric("Net Efficiency",f"{net/1000*100:.1f}%",
                delta=f"{net/1000*100-70:.1f}% vs 70% baseline")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        # Pie chart
        fig_pie = px.pie(
            values=[gwp_steam, gwp_elec, gwp_mea, gwp_const],
            names=['Steam (reboiler)', 'Electricity', 'MEA make-up', 'Construction'],
            color_discrete_sequence=['#EF4444', '#3B82F6', '#8B5CF6', '#6B7280'],
            title=f'LCA Breakdown — {country} grid'
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=380
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        # Country comparison bar
        country_data = {}
        for c in EF:
            gt, nv, *_ = calc_lca(qreb_A, lg, t_abs, p_str, amine, EF[c])
            country_data[c] = {'gwp': gt, 'net': nv, 'eff': nv/1000*100}

        countries   = list(country_data.keys())
        efficiencies= [country_data[c]['eff'] for c in countries]
        bar_cols    = ['#10B981' if country_data[c]['eff'] > 65 else '#EF4444'
                       for c in countries]

        fig_country = go.Figure(go.Bar(
            x=countries, y=efficiencies,
            marker_color=bar_cols,
            text=[f"{e:.1f}%" for e in efficiencies],
            textposition='outside'
        ))
        fig_country.add_hline(y=65, line_dash='dash', line_color='orange',
                              annotation_text='65% minimum target')
        fig_country.update_layout(
            title='Net CO₂ Reduction by Country Grid',
            yaxis_title='Net efficiency (%)',
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=380,
            showlegend=False
        )
        fig_country.update_xaxes(gridcolor='#2E3250')
        fig_country.update_yaxes(gridcolor='#2E3250')
        st.plotly_chart(fig_country, use_container_width=True)

    # LCA insights
    st.divider()
    st.markdown("#### 🌿 Environmental Insights")

    steam_pct = gwp_steam / gwp_tot * 100
    st.markdown(f"""<div class="insight-blue">
    📊 <b>Steam dominates at {steam_pct:.0f}% of GWP.</b> Every 0.5 GJ/t reduction in Q_reboiler
    saves {0.5*1000*56.1/1000:.0f} kg CO₂-eq/t — equivalent to switching from Germany to
    France grid (saves {(EF['Germany']-EF['France'])*(88+12*(p_str-1.5)+12+5*lg):.0f} kg CO₂-eq/t).
    Process optimisation IS environmental optimisation.
    </div>""", unsafe_allow_html=True)

    if net > 700:
        st.markdown(f"""<div class="insight-green">
        ✅ <b>Excellent environmental performance:</b> {net:.0f} kg CO₂-eq avoided per tonne captured.
        Your current conditions achieve {net/1000*100:.1f}% net CO₂ reduction — above the 70% benchmark.
        </div>""", unsafe_allow_html=True)
    elif net > 600:
        st.markdown(f"""<div class="insight-orange">
        🟡 <b>Good but improvable:</b> {net:.0f} kg net avoided. Reducing Q_reboiler by 0.5 GJ/t
        would improve this by ~{0.5*1000*56.1/1000:.0f} kg CO₂-eq/t. Use the Optimisation tab
        to find better operating conditions.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="insight-red">
        ⚠️ <b>High environmental cost:</b> Only {net:.0f} kg net avoided. Your Q_reboiler of
        {qreb_A:.2f} GJ/t is high. Consider lowering absorber temperature, increasing stripper
        pressure, or adjusting L/G toward 2.5–3.5.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — PROCESS INSIGHTS
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 💡 Real-Time Process Recommendations")

    # Generate recommendations based on current inputs
    recommendations = []

    # L/G check
    if lg < 2.0:
        recommendations.append(("🔴 L/G too LOW",
            f"Current L/G={lg:.1f} is below the sweet spot (2.5–3.5). "
            f"Solvent is under-circulated → poor CO₂ regeneration → high Q_reboiler. "
            f"Try increasing L/G toward 2.5.", "red"))
    elif lg > 4.5:
        recommendations.append(("🟡 L/G too HIGH",
            f"Current L/G={lg:.1f} is above the sweet spot (2.5–3.5). "
            f"Too much solvent to heat up → excess sensible heat → high Q_reboiler. "
            f"Try reducing L/G toward 3.0.", "orange"))
    else:
        recommendations.append(("✅ L/G is in the sweet spot",
            f"Current L/G={lg:.1f} is in the optimal range (2.5–3.5). "
            f"Good balance between regeneration quality and sensible heat.", "green"))

    # Temperature check
    if t_abs > 52:
        recommendations.append(("🔴 Absorber temperature HIGH",
            f"Current T={t_abs:.1f}°C reduces CO₂ solubility in MEA (Henry's Law). "
            f"Lower absorber temperature improves absorption. Target: 38–45°C.", "red"))
    elif t_abs < 38:
        recommendations.append(("✅ Absorber temperature OPTIMAL",
            f"Current T={t_abs:.1f}°C is in the ideal range. Good CO₂ solubility.", "green"))
    else:
        recommendations.append(("✅ Absorber temperature GOOD",
            f"Current T={t_abs:.1f}°C is acceptable. Small gains possible by cooling to 40°C.", "green"))

    # Stripper pressure check
    if p_str < 1.5:
        recommendations.append(("🟡 Stripper pressure LOW",
            f"Current P={p_str:.2f} bar. Higher stripper pressure reduces steam requirement. "
            f"Consider increasing toward 1.8–2.0 bar.", "orange"))
    else:
        recommendations.append(("✅ Stripper pressure GOOD",
            f"Current P={p_str:.2f} bar is in a good range for steam efficiency.", "green"))

    # Capture rate check
    if cap < 80:
        recommendations.append(("🔴 Capture rate BELOW target",
            f"Current capture={cap:.1f}%. EU CCS target is ≥90%. "
            f"Increase L/G ratio or amine concentration to improve CO₂ absorption.", "red"))
    elif cap < 85:
        recommendations.append(("🟡 Capture rate BELOW 85% threshold",
            f"Current capture={cap:.1f}%. Below the 85% constraint used in optimisation. "
            f"Consider increasing L/G or amine concentration.", "orange"))
    else:
        recommendations.append(("✅ Capture rate acceptable",
            f"Current capture={cap:.1f}%. Meets the 85% operating target.", "green"))

    # Q_reboiler check
    if qreb_A > 5.0:
        recommendations.append(("🔴 Q_reboiler VERY HIGH",
            f"At {qreb_A:.2f} GJ/t, energy cost is far above the 4.0 GJ/t industry target. "
            f"This is primarily driven by your temperature and L/G settings.", "red"))
    elif qreb_A > 4.0:
        recommendations.append(("🟡 Q_reboiler above target",
            f"At {qreb_A:.2f} GJ/t, you are above the 4.0 GJ/t industry target. "
            f"Small L/G and temperature adjustments can bring this down.", "orange"))
    else:
        recommendations.append(("✅ Q_reboiler within target",
            f"At {qreb_A:.2f} GJ/t, energy cost is at or below the 4.0 GJ/t target. "
            f"Good operating conditions.", "green"))

    # Display recommendations
    for title, body, level in recommendations:
        css_class = f"insight-{level}"
        st.markdown(f"""<div class="{css_class}">
        <b>{title}</b><br>{body}
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Operating condition radar / sensitivity
    st.markdown("#### 📊 How Each Input Affects Q_reboiler")
    st.caption("Sweep each variable ±20% from current value while keeping others fixed")

    sensitivity_data = []
    for var, val, lo, hi, label in [
        ('L/G',    lg,    max(1.0, lg*0.8),    min(5.5, lg*1.2),  'L_G_ratio'),
        ('Temp',   t_abs, max(35,  t_abs-5),   min(60,  t_abs+5), 'absorber_T'),
        ('Amine',  amine, max(20,  amine-5),   min(40,  amine+5), 'amine_conc'),
        ('Pressure',p_str,max(1.3, p_str-0.3), min(2.4, p_str+0.3),'stripper_P'),
    ]:
        inputs_lo = [amine,lg,t_abs,p_str,flue]
        inputs_hi = [amine,lg,t_abs,p_str,flue]
        idx = ['amine_conc','L_G_ratio','absorber_T','stripper_P','flue_CO2'].index(label)
        inputs_lo[idx] = lo
        inputs_hi[idx] = hi
        q_lo = physics_qreb(*inputs_lo[:4]) + float(models['correction'].predict(
               np.array([inputs_lo]))[0])
        q_hi = physics_qreb(*inputs_hi[:4]) + float(models['correction'].predict(
               np.array([inputs_hi]))[0])
        sensitivity_data.append({'Variable': var,
                                  'Low value':  round(q_lo, 3),
                                  'High value': round(q_hi, 3),
                                  'Delta':      round(q_hi - q_lo, 3)})

    df_sens = pd.DataFrame(sensitivity_data)

    fig_sens = go.Figure()
    fig_sens.add_bar(
        x=df_sens['Variable'], y=df_sens['Delta'],
        marker_color=['#EF4444' if d > 0 else '#10B981' for d in df_sens['Delta']],
        text=[f"{d:+.3f}" for d in df_sens['Delta']],
        textposition='outside'
    )
    fig_sens.update_layout(
        title='Change in Q_reboiler when each variable increases by ±20%',
        yaxis_title='ΔQ_reboiler (GJ/t)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300,
        showlegend=False
    )
    fig_sens.update_xaxes(gridcolor='#2E3250')
    fig_sens.update_yaxes(gridcolor='#2E3250', zeroline=True, zerolinecolor='#4B5563')
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption("Red bar = increasing this variable INCREASES Q_reboiler (bad). Green bar = increasing it DECREASES Q_reboiler (good).")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMISATION
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### ⚡ Find Optimal Operating Conditions")
    st.markdown("The optimiser calls the surrogate model ~1000 times in <1 second — replacing 7 days of Aspen Plus simulation.")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_capture = st.slider("Minimum capture rate (%)", 70, 95, 85, 1,
                                help="Optimiser will penalise solutions below this")
    with col2:
        grid_opt = st.selectbox("Grid for LCA optimisation", list(EF.keys()), index=0)
    with col3:
        n_restarts = st.selectbox("Number of random restarts", [10, 20, 30, 50], index=2,
                                  help="More restarts = more likely to find global minimum")

    run_opt = st.button("🚀 Run Optimisation", type="primary", use_container_width=True)

    # Store results in session state so they persist after button click
    if 'opt_result' not in st.session_state:
        st.session_state.opt_result = None

    if run_opt:
        ef_opt = EF[grid_opt]

        with st.spinner(f"Running {n_restarts} optimisation restarts..."):
            def objective(params):
                a, l, t, p, f = params
                X_ = np.array([[a, l, t, p, f]])
                c_ = float(models['capture'].predict(X_)[0])
                q_ = float(models['qreb'].predict(X_)[0])
                e_ = (88 + 12*(p-1.5)) + (12 + 5*l)
                sl_= max(0.3, 0.3 + 0.022*t + 0.016*a)
                gwp= q_*1000*56.1/1000 + e_*ef_opt + sl_*2.49 + 15
                return gwp + max(0, min_capture - c_) * 200

            bounds = [(20,40),(1.0,5.5),(35,58),(1.3,2.4),(7,13)]
            best = None
            progress = st.progress(0)
            for i in range(n_restarts):
                x0  = [np.random.uniform(b[0], b[1]) for b in bounds]
                res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
                if best is None or res.fun < best.fun:
                    best = res
                progress.progress((i+1) / n_restarts)

        # Store result
        o = best.x
        X_opt = np.array([[o[0], o[1], o[2], o[3], o[4]]])
        cap_opt  = float(models['capture'].predict(X_opt)[0])
        qreb_opt = float(models['qreb'].predict(X_opt)[0])
        gwp_opt  = best.fun
        net_opt  = 1000 - gwp_opt

        st.session_state.opt_result = {
            'amine': o[0], 'lg': o[1], 't': o[2], 'p': o[3], 'flue': o[4],
            'cap': cap_opt, 'qreb': qreb_opt,
            'gwp': gwp_opt, 'net': net_opt,
            'grid': grid_opt, 'min_cap': min_capture
        }

    # Display stored result
    if st.session_state.opt_result is not None:
        r = st.session_state.opt_result
        st.divider()
        st.markdown(f"#### ✅ Optimal Conditions Found  *(grid: {r['grid']}, constraint: ≥{r['min_cap']}% capture)*")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Amine",      f"{r['amine']:.1f} wt%")
        c2.metric("L/G",        f"{r['lg']:.2f} mol/mol")
        c3.metric("Absorber T", f"{r['t']:.1f} °C")
        c4.metric("Stripper P", f"{r['p']:.2f} bar")
        c5.metric("Flue CO₂",   f"{r['flue']:.1f} vol%")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capture Rate",    f"{r['cap']:.1f}%",
                  delta=f"{r['cap']-min_capture:.1f}% vs constraint")
        c2.metric("Q_reboiler",      f"{r['qreb']:.3f} GJ/t",
                  delta=f"{r['qreb']-qreb_A:.3f} vs current",
                  delta_color="inverse")
        c3.metric("Min GWP",         f"{r['gwp']:.0f} kg CO₂-eq/t",
                  delta=f"{r['gwp']-gwp_tot:.0f} vs current",
                  delta_color="inverse")
        c4.metric("Net Avoided",     f"{r['net']:.0f} kg CO₂-eq/t",
                  delta=f"{r['net']-net:.0f} vs current")

        st.divider()

        # Compare current vs optimal
        st.markdown("#### 📊 Current vs Optimal")
        params    = ['Amine (wt%)', 'L/G (mol/mol)', 'T (°C)', 'P (bar)']
        current_v = [amine, lg, t_abs, p_str]
        optimal_v = [r['amine'], r['lg'], r['t'], r['p']]

        fig_comp = go.Figure()
        fig_comp.add_bar(name='Current',  x=params, y=current_v,
                         marker_color='#4A90D9', opacity=0.8)
        fig_comp.add_bar(name='Optimal',  x=params, y=optimal_v,
                         marker_color='#10B981', opacity=0.8)
        fig_comp.update_layout(
            barmode='group',
            title='Current operating conditions vs optimal',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=320,
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        fig_comp.update_xaxes(gridcolor='#2E3250')
        fig_comp.update_yaxes(gridcolor='#2E3250')
        st.plotly_chart(fig_comp, use_container_width=True)

        # Insight on the result
        qreb_saving = qreb_A - r['qreb']
        gwp_saving  = gwp_tot - r['gwp']

        if qreb_saving > 0:
            st.markdown(f"""<div class="insight-green">
            ✅ <b>Optimisation found a better operating point.</b><br>
            Reducing Q_reboiler from <b>{qreb_A:.3f}</b> → <b>{r['qreb']:.3f} GJ/t</b>
            saves <b>{qreb_saving:.3f} GJ/t</b> of reboiler duty, which reduces LCA footprint
            by <b>{gwp_saving:.0f} kg CO₂-eq/t</b>. Net CO₂ avoided improves from
            {net:.0f} → <b>{r['net']:.0f} kg/t</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="insight-blue">
            ℹ️ <b>Current conditions are already close to optimal.</b><br>
            The optimiser could only improve GWP by {abs(gwp_saving):.0f} kg CO₂-eq/t
            from your current settings. You are operating near the minimum environmental
            impact point for this grid and capture constraint.
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Literature benchmark:**")
        st.caption("Aker MTU 2022: SRD 3.6–3.8 GJ/t at 90% capture, L/G ≈ 2.5, T_abs ≈ 40°C, 25–30 wt% MEA")
        st.caption("TCM Mongstad 2015 (30 wt% MEA): SRD 3.5–4.5 GJ/t")
        st.caption("DTU WtE Amager Bakke 2021 (35 wt% MEA, WtE flue gas): SRD 3.73–4.18 GJ/t")

    else:
        st.info("👆 Set your constraints above and click **Run Optimisation** to find the best operating conditions.")
        st.markdown("""
        **What the optimiser does:**
        - Tries 30 different random starting points across the full operating range
        - For each starting point, uses gradient descent (L-BFGS-B) to find the local minimum GWP
        - Returns the best result across all 30 runs
        - Penalises solutions where capture rate falls below your constraint
        - Calls the surrogate model ~1000 times — would take 7 days with Aspen Plus

        **Why 30 restarts?**
        The GWP landscape has multiple local minima. One random start might find a local
        minimum that is not the global best. More restarts = higher confidence the true
        global optimum is found.
        """)
