# === co2_dashboard.py ===
import streamlit as st, numpy as np, pandas as pd
import joblib, plotly.express as px, pathlib
from scipy.optimize import minimize

st.set_page_config(page_title='CO₂ Capture', layout='wide')
st.title('CO₂ Capture: Performance + LCA + Optimisation')
st.caption('Real literature-grounded dataset | TCM Mongstad · DTU WtE · Aker MTU')

SCRIPT_DIR = pathlib.Path(__file__).parent

def load():
    return (joblib.load(SCRIPT_DIR/'capture_surrogate.pkl'),
            joblib.load(SCRIPT_DIR/'qreb_surrogate.pkl'),
            joblib.load(SCRIPT_DIR/'qreb_correction_A.pkl'),
            joblib.load(SCRIPT_DIR/'param_model_C.pkl'))

model_cap, model_q, correction_A, param_C = load()
EF = {'Germany':0.40,'France':0.05,'Norway':0.02,'Poland':0.78,'India':0.71}

tab1, tab2, tab3 = st.tabs(['🎯 Predictions','🌿 LCA','⚡ Optimisation'])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Process Inputs')
        amine  = st.slider('MEA (wt%)',       20.0, 40.0, 30.0, 0.5)
        lg     = st.slider('L/G (mol/mol)',    1.0,  5.5,  3.0, 0.1)
        t_abs  = st.slider('Absorber T (°C)', 35.0, 60.0, 45.0, 0.5)
        p_str  = st.slider('Stripper P (bar)', 1.3,  2.4,  1.8, 0.05)
        flue   = st.slider('Flue CO₂ (%)',     7.0, 13.0, 10.0, 0.5)

    X = np.array([[amine, lg, t_abs, p_str, flue]])
    cap    = float(model_cap.predict(X)[0])
    qreb_ml= float(model_q.predict(X)[0])

    # Physics prediction
    Cp,MW,da = 3.5, 44.01, 0.20
    Q_sens   = lg*Cp*(120-t_abs)*MW/(da*1000)/1000
    qreb_ph  = max(2.5, min(Q_sens + 85*1000/MW/1e6 + 0.8/p_str, 8.0))

    # Hybrid A: physics + ML correction
    corr     = float(correction_A.predict(X)[0])
    qreb_A   = qreb_ph + corr

    # Hybrid C: ML parameter
    da_ml    = float(param_C.predict(X)[0])
    da_ml    = np.clip(da_ml, 0.05, 0.50)
    Qs_C     = lg*Cp*(120-t_abs)*MW/(da_ml*1000)/1000
    qreb_C   = max(2.5, min(Qs_C + 85*1000/MW/1e6 + 0.8/p_str, 8.0))

    with c2:
        st.subheader('Predictions (all 4 approaches)')
        r1,r2,r3,r4 = st.columns(4)
        r1.metric('Physics',   f'{qreb_ph:.3f} GJ/t')
        r2.metric('Pure ML',   f'{qreb_ml:.3f} GJ/t')
        r3.metric('Hybrid A',  f'{qreb_A:.3f} GJ/t')
        r4.metric('Hybrid C',  f'{qreb_C:.3f} GJ/t')
        st.metric('Capture Rate', f'{cap:.1f}%')
        if cap >= 85: st.success('✅ Capture target met (≥85%)')
        else:         st.warning('⚠️ Below 85% capture target')
        st.caption(f'Arch C detected delta_alpha = {da_ml:.3f} (physics assumed 0.20)')

with tab2:
    country = st.selectbox('Electricity Grid', list(EF.keys()))
    ef      = EF[country]
    elec    = (88+12*(p_str-1.5))+(12+5*lg)
    sloss   = max(0.3, 0.3+0.022*t_abs+0.016*amine)
    gwp_steam= qreb_A*1000*56.1/1000
    gwp_elec = elec*ef
    gwp_mea  = sloss*2.49
    gwp_tot  = gwp_steam+gwp_elec+gwp_mea+15
    net      = 1000-gwp_tot
    c1,c2,c3 = st.columns(3)
    c1.metric('GWP Total',   f'{gwp_tot:.0f} kg CO₂-eq/t')
    c2.metric('Net Avoided', f'{net:.0f} kg CO₂-eq/t')
    c3.metric('Net Efficiency', f'{net/1000*100:.1f}%')
    fig = px.pie(values=[gwp_steam,gwp_elec,gwp_mea,15],
                 names=['Steam','Electricity','MEA','Construction'],
                 title=f'LCA breakdown | {country} grid')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if st.button('Find optimal conditions (30 restarts)'):
        def obj(p):
            X_  = np.array([[p[0],p[1],p[2],p[3],p[4]]])
            c_  = float(model_cap.predict(X_)[0])
            q_  = float(model_q.predict(X_)[0])
            e_  = (88+12*(p[3]-1.5))+(12+5*p[1])
            sl_ = max(0.3,0.3+0.022*p[2]+0.016*p[0])
            gwp_= q_*1000*56.1/1000+e_*0.40+sl_*2.49+15
            return gwp_ + max(0,85-c_)*50
        bds = [(20,40),(1.0,5.5),(35,58),(1.3,2.4),(7,13)]
        best=None
        for _ in range(30):
            x0=[np.random.uniform(b[0],b[1]) for b in bds]
            r=minimize(obj,x0,bounds=bds,method='L-BFGS-B')
            if best is None or r.fun<best.fun: best=r
        o=best.x

        st.success(f'Optimal: amine={o[0]:.1f} wt%, L/G={o[1]:.2f}, T={o[2]:.1f}°C, P={o[3]:.2f} bar')
        st.metric('Min GWP', f'{best.fun:.0f} kg CO₂-eq/t')
        st.metric('Net avoided', f'{1000-best.fun:.0f} kg CO₂-eq/t')