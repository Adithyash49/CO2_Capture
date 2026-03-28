# === CO2_Capture_Hybrid.py ===
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pathlib
from scipy.optimize import minimize

st.set_page_config(page_title='CO₂ Capture', page_icon='🏭', layout='wide')

st.markdown("""
<style>
.insight-green{background:#0D2B1F;border-left:4px solid #00C875;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:14px;}
.insight-orange{background:#2B1A0D;border-left:4px solid #FF8C00;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:14px;}
.insight-red{background:#2B0D0D;border-left:4px solid #FF4444;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:14px;}
.insight-blue{background:#0D1A2B;border-left:4px solid #4A90D9;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:14px;}
</style>""", unsafe_allow_html=True)

st.title('🏭 CO₂ Capture: Hybrid Model Dashboard')
st.caption('Real literature-grounded dataset · TCM Mongstad · DTU WtE · Aker MTU · 300 Aspen Plus runs')
st.divider()

SCRIPT_DIR = pathlib.Path(__file__).parent

@st.cache_resource
def load():
    return (
        joblib.load(SCRIPT_DIR / 'capture_surrogate.pkl'),
        joblib.load(SCRIPT_DIR / 'qreb_surrogate.pkl'),
        joblib.load(SCRIPT_DIR / 'qreb_correction_A.pkl'),
        joblib.load(SCRIPT_DIR / 'param_model_C.pkl'),
    )

model_cap, model_q, correction_A, param_C = load()
EF = {'Germany':0.40,'France':0.05,'Norway':0.02,'Poland':0.78,'India':0.71}

def physics_qreb(amine, lg, t_abs, p_str):
    Q = -2.402 + 0.849/lg + 0.039*lg**2 + 0.120*t_abs + 0.723/p_str
    return float(np.clip(Q, 2.5, 8.0))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Process Inputs")
    amine = st.slider('MEA Concentration (wt%)',  20.0, 40.0, 30.0, 0.5,
                      help='30 wt% = industry standard.')
    lg    = st.slider('L/G Ratio (mol/mol)',       1.0,  5.5,  3.0, 0.1,
                      help='Sweet spot 2.5–3.5.')
    t_abs = st.slider('Absorber Temperature (°C)', 35.0, 60.0, 45.0, 0.5,
                      help='Optimum 40–45°C.')
    p_str = st.slider('Stripper Pressure (bar)',   1.3,  2.4,  1.8, 0.05,
                      help='Higher = less steam needed.')
    flue  = st.slider('Flue Gas CO₂ (%)',          7.0, 13.0, 10.0, 0.5,
                      help='WtE flue gas: 8–12 vol%.')
    st.divider()
    st.markdown("### 🌿 LCA Settings")
    country = st.selectbox('Electricity Grid', list(EF.keys()))
    ef      = EF[country]
    st.divider()
    st.caption("Physics R²=0.82 → Hybrid A R²=0.97")
    st.caption("Dataset: CO2_Capture_Dataset_v3.xlsx")

# ── Compute all predictions ───────────────────────────────────────────────────
X       = np.array([[amine, lg, t_abs, p_str, flue]])
cap     = float(model_cap.predict(X)[0])
qreb_ml = float(model_q.predict(X)[0])
qreb_ph = physics_qreb(amine, lg, t_abs, p_str)
corr    = float(correction_A.predict(X)[0])
qreb_A  = qreb_ph + corr

# Architecture C — widen clip so it doesn't stick at minimum
da_raw  = float(param_C.predict(X)[0])
da_ml   = float(np.clip(da_raw, 0.10, 0.50))   # widened from 0.05 → 0.10
Cp, MW  = 3.5, 44.01
Qs_C    = lg*Cp*(120-t_abs)*MW/(da_ml*1000)/1000
qreb_C  = float(np.clip(Qs_C+85*1000/MW/1e6+0.8/p_str, 2.5, 8.0))

# LCA
elec      = (88+12*(p_str-1.5))+(12+5*lg)
sloss     = max(0.3, 0.3+0.022*t_abs+0.016*amine)
gwp_steam = qreb_A*1000*56.1/1000
gwp_elec  = elec*ef
gwp_mea   = sloss*2.49
gwp_const = 15.0
gwp_tot   = gwp_steam+gwp_elec+gwp_mea+gwp_const
net       = 1000-gwp_tot

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs([
    '🎯 Predictions',
    '🌿 LCA & Environment',
    '💡 Insights',
    '⚡ Optimisation'
])


# ══════════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ══════════════════════════════════════════════
with tab1:
    # KPIs — 2 decimal places
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Capture Rate", f"{cap:.2f}%",
              delta=f"{cap-85:.2f}% vs 85% target")
    c2.metric("⚛️ Physics",   f"{qreb_ph:.2f} GJ/t", help="Simplified Oexmann R²=0.82")
    c3.metric("🤖 Pure ML",   f"{qreb_ml:.2f} GJ/t", help="XGBoost surrogate R²=0.95")
    c4.metric("🔀 Hybrid A",  f"{qreb_A:.2f} GJ/t",  help="Physics + ML correction R²=0.97")
    c5.metric("⚙️ Hybrid C",  f"{qreb_C:.2f} GJ/t",  help="ML-predicted working capacity")

    if cap >= 85:
        st.success(f'✅ Capture target met: {cap:.2f}% ≥ 85%')
    else:
        st.warning(f'⚠️ Below 85% target — current: {cap:.2f}%')

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### All 4 Models vs Industry Target")
        names = ['Physics\n(R²=0.82)','Pure ML\n(R²=0.95)',
                 'Hybrid A\n(R²=0.97)','Hybrid C\n(R²=0.97)']
        vals  = [qreb_ph, qreb_ml, qreb_A, qreb_C]
        cols  = ['#6B7280','#4A90D9','#F97316','#10B981']
        fig_b = go.Figure()
        for n,v,c in zip(names,vals,cols):
            fig_b.add_bar(x=[n],y=[v],marker_color=c,
                          text=[f'{v:.2f}'],textposition='outside')
        fig_b.add_hline(y=4.0,line_dash='dash',line_color='red',
                        annotation_text='Industry target 4.0 GJ/t')
        fig_b.update_layout(
            yaxis_title='Q_reboiler (GJ/t CO₂)',showlegend=False,
            yaxis_range=[0,max(vals)*1.3],
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',height=360)
        fig_b.update_xaxes(gridcolor='#333')
        fig_b.update_yaxes(gridcolor='#333')
        st.plotly_chart(fig_b,use_container_width=True)

    with right:
        st.markdown("#### U-Shape: Q_reboiler vs L/G (your conditions)")
        sweep = np.linspace(1.0,5.5,60)
        ph_s  = [physics_qreb(amine,l,t_abs,p_str) for l in sweep]
        ml_s  = [float(model_q.predict(np.array([[amine,l,t_abs,p_str,flue]]))[0])
                 for l in sweep]
        ha_s  = [physics_qreb(amine,l,t_abs,p_str) +
                 float(correction_A.predict(np.array([[amine,l,t_abs,p_str,flue]]))[0])
                 for l in sweep]
        fig_u = go.Figure()
        fig_u.add_scatter(x=sweep,y=ph_s,name='Physics',
                          line=dict(color='#6B7280',width=2,dash='dot'))
        fig_u.add_scatter(x=sweep,y=ml_s,name='Pure ML',
                          line=dict(color='#4A90D9',width=2))
        fig_u.add_scatter(x=sweep,y=ha_s,name='Hybrid A',
                          line=dict(color='#F97316',width=2.5))
        fig_u.add_vline(x=lg,line_dash='dash',line_color='white',
                        annotation_text=f'Current L/G={lg}',
                        annotation_font_color='white')
        fig_u.add_hline(y=4.0,line_dash='dash',line_color='red',opacity=0.4)
        fig_u.update_layout(
            xaxis_title='L/G (mol/mol)',yaxis_title='Q_reboiler (GJ/t)',
            title='Sweet spot L/G = 2.5–3.5',
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',height=360,legend=dict(bgcolor='rgba(0,0,0,0)'))
        fig_u.update_xaxes(gridcolor='#333')
        fig_u.update_yaxes(gridcolor='#333')
        st.plotly_chart(fig_u,use_container_width=True)

    # Architecture C insight
    st.divider()
    st.markdown("#### 🔍 Architecture C — What ML discovered about the solvent")
    a1,a2,a3 = st.columns(3)
    a1.metric("Physics assumed delta_α","0.20 (fixed)")
    a2.metric("ML predicted delta_α",f"{da_ml:.2f}")
    a3.metric("Difference",f"{da_ml-0.20:+.2f}",delta_color="off")

    if da_ml > 0.22:
        st.markdown(
            f'<div class="insight-green">✅ <b>Solvent performing BETTER than assumed.</b>'
            f' Real working capacity ({da_ml:.2f}) &gt; assumed (0.20).'
            f' Each mole of MEA absorbs more CO₂ than expected.'
            f' Physics would overestimate Q_reboiler — Hybrid C gives the accurate lower value.</div>',
            unsafe_allow_html=True)
    elif da_ml < 0.18:
        st.markdown(
            f'<div class="insight-red">⚠️ <b>Solvent performing WORSE than assumed.</b>'
            f' Real working capacity ({da_ml:.2f}) &lt; assumed (0.20).'
            f' Possible causes: high absorber temperature degrading MEA, solvent near loading limit.'
            f' Real Q_reboiler is higher than physics predicts.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="insight-blue">ℹ️ <b>Working capacity close to assumed value.</b>'
            f' delta_α = {da_ml:.2f} ≈ assumed 0.20.'
            f' Physics and data agree well at these conditions.</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — LCA
# ══════════════════════════════════════════════
with tab2:
    k1,k2,k3 = st.columns(3)
    k1.metric("GWP Total",      f"{gwp_tot:.0f} kg CO₂-eq/t")
    k2.metric("Net Avoided",    f"{net:.0f} kg CO₂-eq/t")
    k3.metric("Net Efficiency", f"{net/1000*100:.1f}%",
              delta=f"{net/1000*100-70:.1f}% vs 70% baseline")
    st.divider()
    left,right = st.columns(2)

    with left:
        fig_pie = px.pie(
            values=[gwp_steam,gwp_elec,gwp_mea,gwp_const],
            names=['Steam (reboiler)','Electricity','MEA make-up','Construction'],
            color_discrete_sequence=['#EF4444','#3B82F6','#8B5CF6','#6B7280'],
            title=f'GWP Breakdown — {country} grid')
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              font_color='white',height=370)
        st.plotly_chart(fig_pie,use_container_width=True)

    with right:
        c_names,c_effs = [],[]
        for c in EF:
            g_c = qreb_A*1000*56.1/1000+elec*EF[c]+sloss*2.49+15
            c_names.append(c)
            c_effs.append((1000-g_c)/1000*100)
        fig_cc = go.Figure(go.Bar(
            x=c_names,y=c_effs,
            marker_color=['#10B981' if e>65 else '#EF4444' for e in c_effs],
            text=[f'{e:.1f}%' for e in c_effs],textposition='outside'))
        fig_cc.add_hline(y=65,line_dash='dash',line_color='orange',
                         annotation_text='65% minimum')
        fig_cc.update_layout(
            title='Net CO₂ Reduction by Country Grid',
            yaxis_title='Net efficiency (%)',yaxis_range=[0,100],
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',height=370,showlegend=False)
        fig_cc.update_xaxes(gridcolor='#333')
        fig_cc.update_yaxes(gridcolor='#333')
        st.plotly_chart(fig_cc,use_container_width=True)

    st.divider()
    steam_pct = gwp_steam/gwp_tot*100
    st.markdown(
        f'<div class="insight-blue">📊 <b>Steam accounts for {steam_pct:.0f}% of GWP.</b>'
        f' Every 0.5 GJ/t reduction in Q_reboiler saves {0.5*1000*56.1/1000:.0f} kg CO₂-eq/t'
        f' — more impact than switching Germany → France grid'
        f' ({(EF["Germany"]-EF["France"])*elec:.0f} kg/t).'
        f' Process optimisation IS environmental optimisation.</div>',
        unsafe_allow_html=True)

    if net > 700:
        st.markdown(
            f'<div class="insight-green">✅ <b>Excellent:</b> {net:.0f} kg CO₂-eq avoided/t'
            f' — {net/1000*100:.1f}% net reduction. Above 70% benchmark.</div>',
            unsafe_allow_html=True)
    elif net > 600:
        st.markdown(
            f'<div class="insight-orange">🟡 <b>Good but improvable:</b> {net:.0f} kg net avoided.'
            f' Reducing Q_reboiler by 0.5 GJ/t adds {0.5*1000*56.1/1000:.0f} kg CO₂-eq/t.'
            f' Use the Optimisation tab.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="insight-red">⚠️ <b>High environmental cost:</b> Only {net:.0f} kg net avoided.'
            f' Q_reboiler = {qreb_A:.2f} GJ/t is high.'
            f' Lower absorber temperature, increase stripper pressure,'
            f' or adjust L/G toward 2.5–3.5.</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### 💡 Real-Time Process Recommendations")
    checks = [
        (lg < 2.0,
         "🔴 L/G too LOW",
         f"L/G={lg:.1f} under-circulates solvent → poor regeneration → high Q_reboiler. Increase toward 2.5.",
         "red"),
        (lg > 4.5,
         "🟡 L/G too HIGH",
         f"L/G={lg:.1f} heats too much solvent → excess sensible heat → high Q_reboiler. Reduce toward 3.0.",
         "orange"),
        (2.0 <= lg <= 4.5,
         "✅ L/G in sweet spot",
         f"L/G={lg:.1f} is in the optimal range 2.0–4.5. Good balance between regeneration and sensible heat.",
         "green"),
        (t_abs > 52,
         "🔴 Absorber temperature HIGH",
         f"T={t_abs:.1f}°C reduces CO₂ solubility in MEA (Henry's Law). Lower toward 38–45°C.",
         "red"),
        (t_abs <= 52,
         "✅ Absorber temperature acceptable",
         f"T={t_abs:.1f}°C is acceptable. Best performance at 38–45°C.",
         "green"),
        (p_str < 1.5,
         "🟡 Stripper pressure LOW",
         f"P={p_str:.2f} bar. Higher pressure reduces steam requirement. Try 1.8–2.0 bar.",
         "orange"),
        (p_str >= 1.5,
         "✅ Stripper pressure good",
         f"P={p_str:.2f} bar is in a good range for steam efficiency.",
         "green"),
        (cap < 80,
         "🔴 Capture rate LOW",
         f"Capture={cap:.2f}%. EU CCS target ≥90%. Increase L/G or amine concentration.",
         "red"),
        (80 <= cap < 85,
         "🟡 Capture below 85% constraint",
         f"Capture={cap:.2f}%. Below 85% operating constraint. Consider increasing L/G or amine.",
         "orange"),
        (cap >= 85,
         "✅ Capture meets target",
         f"Capture={cap:.2f}% meets the 85% operating constraint.",
         "green"),
        (qreb_A > 5.0,
         "🔴 Q_reboiler VERY HIGH",
         f"Q={qreb_A:.2f} GJ/t far above 4.0 GJ/t industry target. Adjust temperature and L/G.",
         "red"),
        (4.0 < qreb_A <= 5.0,
         "🟡 Q_reboiler above target",
         f"Q={qreb_A:.2f} GJ/t above 4.0. Small L/G and temperature adjustments can bring this down.",
         "orange"),
        (qreb_A <= 4.0,
         "✅ Q_reboiler within target",
         f"Q={qreb_A:.2f} GJ/t is at or below the 4.0 GJ/t industry target.",
         "green"),
    ]
    for cond,title,msg,level in checks:
        if cond:
            st.markdown(
                f'<div class="insight-{level}"><b>{title}</b><br>{msg}</div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Which Input Affects Q_reboiler Most?")
    st.caption("Change in Q_reboiler when each variable increases by +10% from current value")

    base      = [amine, lg, t_abs, p_str, flue]
    b_max     = [40.0, 5.5, 60.0, 2.4, 13.0]
    b_min     = [20.0, 1.0, 35.0, 1.3,  7.0]
    var_names = ['L/G', 'Temp', 'Amine', 'Pressure', 'Flue CO₂']
    var_idx   = [1, 2, 0, 3, 4]
    deltas, labels = [], []
    for label, idx in zip(var_names, var_idx):
        hi = min(base[idx]*1.1, b_max[idx])
        lo = max(base[idx]*0.9, b_min[idx])
        inp_hi = base.copy(); inp_hi[idx] = hi
        inp_lo = base.copy(); inp_lo[idx] = lo
        q_hi = (physics_qreb(inp_hi[0],inp_hi[1],inp_hi[2],inp_hi[3]) +
                float(correction_A.predict(np.array([inp_hi]))[0]))
        q_lo = (physics_qreb(inp_lo[0],inp_lo[1],inp_lo[2],inp_lo[3]) +
                float(correction_A.predict(np.array([inp_lo]))[0]))
        deltas.append(round(q_hi-q_lo, 2))
        labels.append(label)

    fig_s = go.Figure(go.Bar(
        x=labels, y=deltas,
        marker_color=['#EF4444' if d>0 else '#10B981' for d in deltas],
        text=[f'{d:+.2f}' for d in deltas], textposition='outside'))
    fig_s.update_layout(
        title='ΔQ_reboiler per ±10% change in each variable',
        yaxis_title='ΔQ_reboiler (GJ/t)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=320, showlegend=False)
    fig_s.update_xaxes(gridcolor='#333')
    fig_s.update_yaxes(gridcolor='#333', zeroline=True, zerolinecolor='#555')
    st.plotly_chart(fig_s, use_container_width=True)
    st.caption("🔴 Red = increasing this variable INCREASES Q_reboiler (costs more energy).  "
               "🟢 Green = increasing it DECREASES Q_reboiler (saves energy).")


# ══════════════════════════════════════════════
# TAB 4 — OPTIMISATION (fast version)
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### ⚡ Find Optimal Operating Conditions")
    st.info("Calls the surrogate model to minimise GWP — far faster than running Aspen Plus.")

    co1, co2, co3 = st.columns(3)
    with co1:
        min_cap_c = st.slider("Minimum capture rate (%)", 70, 95, 85, 1,
                              help="Optimiser penalises solutions below this threshold")
    with co2:
        grid_opt = st.selectbox("Grid for optimisation", list(EF.keys()), index=0)
    with co3:
        # FREE NUMBER INPUT — any value the user wants
        n_starts = st.number_input("Random restarts", min_value=1, max_value=200,
                                   value=15, step=5,
                                   help="More restarts = better chance of global minimum. "
                                        "5–15 is fast, 50+ is thorough.")

    run_btn = st.button("🚀 Run Optimisation", type="primary", use_container_width=True)

    if 'opt_result' not in st.session_state:
        st.session_state.opt_result = None

    if run_btn:
        ef_opt = EF[grid_opt]

        # Objective function — minimise GWP with capture penalty
        def objective(params):
            a, l, t, p, f = params
            X_ = np.array([[a, l, t, p, f]])
            c_ = float(model_cap.predict(X_)[0])
            q_ = float(model_q.predict(X_)[0])
            e_ = (88 + 12*(p-1.5)) + (12 + 5*l)
            sl_= max(0.3, 0.3 + 0.022*t + 0.016*a)
            gwp= q_*1000*56.1/1000 + e_*ef_opt + sl_*2.49 + 15
            return gwp + max(0, min_cap_c - c_) * 200

        bds  = [(20,40),(1.0,5.5),(35,58),(1.3,2.4),(7,13)]
        best = None
        prog = st.progress(0, text="Starting...")

        for i in range(n_starts):
            x0 = [np.random.uniform(b[0], b[1]) for b in bds]
            # maxiter=100 per restart keeps each run fast
            res = minimize(objective, x0=x0, bounds=bds, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-6})
            if best is None or res.fun < best.fun:
                best = res
            prog.progress((i+1)/n_starts,
                          text=f"Restart {i+1}/{n_starts} — best GWP so far: {best.fun:.0f} kg/t")

        prog.empty()

        o      = best.x
        X_o    = np.array([[o[0],o[1],o[2],o[3],o[4]]])
        cap_o  = float(model_cap.predict(X_o)[0])
        qreb_o = float(model_q.predict(X_o)[0])

        st.session_state.opt_result = {
            'amine': o[0], 'lg': o[1], 't': o[2], 'p': o[3], 'flue': o[4],
            'cap': cap_o, 'qreb': qreb_o,
            'gwp': best.fun, 'net': 1000-best.fun,
            'grid': grid_opt
        }

    # Display stored result — survives slider interactions
    if st.session_state.opt_result is not None:
        r = st.session_state.opt_result
        st.divider()
        st.markdown(f"#### ✅ Optimal Conditions  *(grid: {r['grid']}, capture ≥ {min_cap_c}%)*")

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Amine",      f"{r['amine']:.1f} wt%",
                  delta=f"{r['amine']-amine:+.1f} vs current", delta_color="off")
        d2.metric("L/G",        f"{r['lg']:.2f} mol/mol",
                  delta=f"{r['lg']-lg:+.2f} vs current",       delta_color="off")
        d3.metric("Absorber T", f"{r['t']:.1f} °C",
                  delta=f"{r['t']-t_abs:+.1f} vs current",     delta_color="off")
        d4.metric("Stripper P", f"{r['p']:.2f} bar",
                  delta=f"{r['p']-p_str:+.2f} vs current",     delta_color="off")
        d5.metric("Flue CO₂",   f"{r['flue']:.1f} vol%",
                  delta=f"{r['flue']-flue:+.1f} vs current",   delta_color="off")

        st.divider()
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Capture Rate",  f"{r['cap']:.2f}%",
                  delta=f"{r['cap']-cap:+.2f}%")
        e2.metric("Q_reboiler",    f"{r['qreb']:.2f} GJ/t",
                  delta=f"{r['qreb']-qreb_A:+.2f}",  delta_color="inverse")
        e3.metric("Min GWP",       f"{r['gwp']:.0f} kg CO₂-eq/t",
                  delta=f"{r['gwp']-gwp_tot:+.0f}",  delta_color="inverse")
        e4.metric("Net Avoided",   f"{r['net']:.0f} kg CO₂-eq/t",
                  delta=f"{r['net']-net:+.0f}")

        # Current vs optimal bar chart
        st.divider()
        st.markdown("#### 📊 Current vs Optimal")
        params    = ['Amine (wt%)','L/G (mol/mol)','T (°C)','P (bar)']
        curr_v    = [amine, lg, t_abs, p_str]
        opt_v     = [r['amine'], r['lg'], r['t'], r['p']]
        fig_cmp   = go.Figure()
        fig_cmp.add_bar(name='Current', x=params, y=curr_v, marker_color='#4A90D9',
                        opacity=0.8, text=[f'{v:.2f}' for v in curr_v],
                        textposition='outside')
        fig_cmp.add_bar(name='Optimal', x=params, y=opt_v, marker_color='#10B981',
                        opacity=0.8, text=[f'{v:.2f}' for v in opt_v],
                        textposition='outside')
        fig_cmp.update_layout(
            barmode='group', title='Current vs Optimal operating point',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=330, legend=dict(bgcolor='rgba(0,0,0,0)'))
        fig_cmp.update_xaxes(gridcolor='#333')
        fig_cmp.update_yaxes(gridcolor='#333')
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Insight on result
        q_saving   = qreb_A - r['qreb']
        gwp_saving = gwp_tot - r['gwp']
        if q_saving > 0.05:
            st.markdown(
                f'<div class="insight-green">✅ <b>Better operating point found.</b><br>'
                f'Q_reboiler: {qreb_A:.2f} → <b>{r["qreb"]:.2f} GJ/t</b> '
                f'(saves {q_saving:.2f} GJ/t).<br>'
                f'GWP: {gwp_tot:.0f} → <b>{r["gwp"]:.0f} kg CO₂-eq/t</b> '
                f'(saves {gwp_saving:.0f} kg/t).<br>'
                f'Net avoided: {net:.0f} → <b>{r["net"]:.0f} kg/t</b>.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="insight-blue">ℹ️ <b>Current conditions are already near optimal.</b> '
                f'Optimiser improved GWP by only {abs(gwp_saving):.0f} kg CO₂-eq/t. '
                f'You are close to the minimum environmental impact point.</div>',
                unsafe_allow_html=True)

        st.divider()
        st.markdown("**Literature benchmarks:**")
        st.caption("Aker MTU 2022: Q_reb 3.6–3.8 at 90% · L/G≈2.5 · T≈40°C · 25–30 wt% MEA")
        st.caption("TCM Mongstad 2015 (30 wt% MEA): Q_reb 3.5–4.5 GJ/t")
        st.caption("DTU WtE Amager Bakke 2021 (35 wt% MEA): Q_reb 3.73–4.18 GJ/t")

    else:
        st.markdown("""
        👆 Set constraints above and click **Run Optimisation**.

        **Speed guide:**
        | Restarts | Time | Use when |
        |---|---|---|
        | 5–10 | < 1 second | Quick exploration |
        | 15–20 | 1–2 seconds | Normal use |
        | 50–100 | 5–10 seconds | Thorough search |
        | 200 | ~20 seconds | Maximum confidence |

        **Why results persist:** stored in session state — moving sliders will NOT clear results.
        Click Run again only if you want a fresh search with new constraints.
        """)
