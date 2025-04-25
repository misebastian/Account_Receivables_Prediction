# -----------------------------------------------------------
# Streamlit – EDA interactivo + predicción DaysLate (XGB)
# 25-abr-2025 – carga modelo .pkl (no re-entrena)
# -----------------------------------------------------------

import warnings, json, joblib
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────  configuración  ──────────────────────────
st.set_page_config(page_title="📈 AR Prediction", page_icon="💰", layout="centered")
st.markdown("""
<style>
:root { --primary-color:#2a9df4; --text-color:#e0e0e0; }
footer, #MainMenu {visibility:hidden;}
/* reduce ancho del sidebar */
section[data-testid="stSidebar"] > div:first-child {width:240px;}
/* tablas oscuras */
thead tr th {background-color:#111!important; color:#e0e0e0!important;}
</style>
""", unsafe_allow_html=True)
warnings.filterwarnings("ignore")

# ─────────────────────────  rutas de ficheros  ────────────────────────
DATA_FILE   = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE  = Path("xgb_model.pkl")               # ya entrenado
META_FILE   = Path("model_meta.json")             # opcional

# ────────────────────────────  carga datos  ───────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    # codificación mínima
    df["Disputed"]      = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})
    df["countryCode"]   = df["countryCode"].astype("category")
    return df

if not DATA_FILE.exists():
    st.error("❌ No se encontró el Excel de cuentas por cobrar.")
    st.stop()

raw_df = load_data(DATA_FILE)
num_cols = ["InvoiceAmount", "DaysToSettle", "DaysLate"]

# ──────────────────────────────  EDA  ────────────────────────────────
st.title("Exploratorio de cuentas por cobrar")

kpi1, kpi2 = st.columns(2)
kpi1.metric("Media DaysLate",  f"{raw_df['DaysLate'].mean():.2f}")
kpi2.metric("Desv. típica",   f"{raw_df['DaysLate'].std():.2f}")

with st.expander("📋  Tabla descriptiva"):
    st.dataframe(raw_df[num_cols + ["Disputed","PaperlessBill"]].describe().T.round(2), use_container_width=True)

# variable a estudiar
sel_var = st.sidebar.selectbox("Variable numérica", num_cols, index=0)

st.subheader(f"Distribución de **{sel_var}**")

col_h, col_b = st.columns(2)
col_h.plotly_chart(px.histogram(raw_df, x=sel_var, nbins=35,
                                color_discrete_sequence=["#2a9df4"],
                                title="Histograma"), use_container_width=True)

col_b.plotly_chart(px.box(raw_df, y=sel_var,
                          color_discrete_sequence=["#e74c3c"],
                          title="Boxplot"), use_container_width=True)

# matriz de correlación
with st.expander("🔥 Matriz de correlación"):
    corr = raw_df[num_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values,
                                    x=corr.columns, y=corr.columns,
                                    colorscale="RdBu_r", zmin=-1, zmax=1))
    fig_corr.update_layout(height=600, margin=dict(l=40,r=40,b=40,t=40))
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ──────────────────────  carga modelo entrenado  ──────────────────────
if not MODEL_FILE.exists():
    st.error("❌ No se encontró el modelo entrenado (.pkl).")
    st.stop()

model = joblib.load(MODEL_FILE)

# métricas entrenadas (si existen)
if META_FILE.exists():
    with open(META_FILE) as fp:
        meta = json.load(fp)
    st.caption(f"Modelo XGB ▸ MAE {meta['MAE']}  |  RMSE {meta['RMSE']}  | R² {meta['R2']}")
else:
    st.caption("Modelo XGB cargado.")

# ──────────────────────────  predicción  ─────────────────────────────
st.header("🧮 Predicción de **DaysLate**")

# opciones para el formulario
country_map = {cat:i for i,cat in enumerate(raw_df["countryCode"].cat.categories)}

with st.form("form"):
    c1,c2 = st.columns(2)
    with c1:
        country = st.selectbox("countryCode", list(country_map.keys()))
        inv_amt = st.number_input("InvoiceAmount", 0.0, 10000.0, value=60.0, step=1.0)
        disputed  = st.selectbox("Disputed", ["No","Yes"])
    with c2:
        paperless = st.selectbox("PaperlessBill", ["Paper","Electronic"])
        days_sett = st.number_input("DaysToSettle", 0, 120, value=30, step=1)

    sent = st.form_submit_button("Predecir")

if sent:
    row = pd.DataFrame([{
        "countryCode"  : country,
        "InvoiceAmount": inv_amt,
        "Disputed"     : 1 if disputed=="Yes" else 0,
        "PaperlessBill": 1 if paperless=="Electronic" else 0,
        "DaysToSettle" : days_sett
    }])
    # convertir country a código numérico según el mapeo usado al entrenar
    row["countryCode"] = country_map[country]
    pred = model.predict(row)[0]

    if pred <= 0:
        st.success(f"✅ Pago estimado **{abs(pred):.1f} días ANTES** del vencimiento.")
    else:
        st.error  (f"🚨 Retraso estimado de **{pred:.1f} días**.")

st.caption("© 2025 – Demo Accounts Receivable Prediction")
