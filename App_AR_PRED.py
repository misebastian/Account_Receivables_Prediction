# ──────────────────────────────────────────────────────────────
#  App_AR_PRED.py   ·   Streamlit v4  (compact + robust)
# ──────────────────────────────────────────────────────────────
import warnings, json, io, base64, datetime as dt
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AR Predictor", page_icon="📈", layout="wide")

# ╔════════════════════════════════════════════════════════════╗
#  1 · Utilidades de carga ────────────────────────────────────
# ╚════════════════════════════════════════════════════════════╝
DATA_FILE   = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE  = Path("ar_pipeline.pkl")

@st.cache_data(show_spinner=False)
def load_raw(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})
    for c in ["InvoiceDate", "DueDate"]:
        df[c] = pd.to_datetime(df[c])
        df[[f"{c}_year", f"{c}_month", f"{c}_day"]] = np.c_[
            df[c].dt.year, df[c].dt.month, df[c].dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_pipeline(fp: Path):
    return joblib.load(fp)

if not DATA_FILE.exists() or not MODEL_FILE.exists():
    st.error("🚫 Falta el Excel o el modelo *.pkl* en el repositorio.")
    st.stop()

df_raw      = load_raw(DATA_FILE)
pipeline    = load_pipeline(MODEL_FILE)

# Características usadas en el entrenamiento (exactamente las mismas)
FEATS = ["countryCode","InvoiceAmount","Disputed","PaperlessBill","DaysToSettle",
         "InvoiceDate_year","InvoiceDate_month","InvoiceDate_day",
         "DueDate_year","DueDate_month","DueDate_day"]

# ╔════════════════════════════════════════════════════════════╗
#  2 · KPI compactos ──────────────────────────────────────────
# ╚════════════════════════════════════════════════════════════╝
st.markdown(
    """
    <style>
    .small-card {padding:0.25rem 0.5rem !important}
    .metric .metric-label {font-size:0.75rem;color:gray}
    .metric .metric-value {font-size:1.1rem}
    section[data-testid="stSidebar"] > div:first-child {width:240px}
    #MainMenu, footer {visibility:hidden}
    </style>
    """, unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Facturas", f"{len(df_raw):,}")
k2.metric("Clientes únicos", f"{df_raw['customerID'].nunique():,}")
k3.metric("Retraso medio", f"{df_raw['DaysLate'].mean():.1f} d")
k4.metric("% en fecha",
          f"{(df_raw['DaysLate']<=0).mean()*100:.1f} %")

# ╔════════════════════════════════════════════════════════════╗
#  3 · EDA rápido (selector lateral) ──────────────────────────
# ╚════════════════════════════════════════════════════════════╝
numeric_cols = ["InvoiceAmount","DaysLate","DaysToSettle"]
eda_var = st.sidebar.selectbox("Variable numérica", numeric_cols)

st.subheader(f"Distribución de **{eda_var}**", divider="gray")
hcol, bcol = st.columns(2)
hcol.plotly_chart(
    px.histogram(df_raw, x=eda_var, nbins=40, height=250,
                 color_discrete_sequence=["#3498db"]),
    use_container_width=True)
bcol.plotly_chart(
    px.box(df_raw, y=eda_var, height=250,
           color_discrete_sequence=["#e74c3c"]),
    use_container_width=True)

with st.expander("💡  Estadísticas rápidas"):
    st.dataframe(df_raw[eda_var].describe().to_frame().T.round(2), height=80)

# ╔════════════════════════════════════════════════════════════╗
#  4 · Heatmap de correlación mini ────────────────────────────
# ╚════════════════════════════════════════════════════════════╝
with st.expander("🔗  Correlación básica"):
    corr = df_raw[numeric_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Viridis", zmin=-1, zmax=1))
    fig.update_layout(height=300, margin=dict(l=30,r=30,b=20,t=20))
    st.plotly_chart(fig, use_container_width=True)

# ╔════════════════════════════════════════════════════════════╗
#  5 · Formulario de predicción ───────────────────────────────
# ╚════════════════════════════════════════════════════════════╝
st.header("🎯 Predicción de DaysLate", divider="gray")

with st.form("pred"):
    c1, c2 = st.columns(2)
    cc  = c1.selectbox("countryCode", sorted(df_raw["countryCode"].unique()))
    amt = c1.number_input("InvoiceAmount", 0.0, 1e6, step=10.0, value=50.0)
    disp= c1.selectbox("Disputed", ["No","Yes"])
    pbl = c1.selectbox("PaperlessBill", ["Paper","Electronic"])
    dts = c1.number_input("DaysToSettle", 0, 120, step=1, value=30)

    inv_date = c2.date_input("InvoiceDate", value=dt.date(2013,9,1))
    due_date = c2.date_input("DueDate",    value=dt.date(2013,10,1))

    ok = st.form_submit_button("Predecir")

if ok:
    row = {
        "countryCode":       cc,
        "InvoiceAmount":     amt,
        "Disputed":          1 if disp=="Yes" else 0,
        "PaperlessBill":     1 if pbl=="Electronic" else 0,
        "DaysToSettle":      dts,
        "InvoiceDate_year":  inv_date.year,
        "InvoiceDate_month": inv_date.month,
        "InvoiceDate_day":   inv_date.day,
        "DueDate_year":      due_date.year,
        "DueDate_month":     due_date.month,
        "DueDate_day":       due_date.day,
    }
    X_new = pd.DataFrame([row])[FEATS]
    pred  = float(pipeline.predict(X_new)[0])

    if pred <= 0:
        st.success(f"✅ Pago estimado **{abs(pred):.1f} d antes** del vencimiento.")
    else:
        st.error(f"🚨 Retraso estimado de **{pred:.1f} días**.")

st.caption("© 2025 – Demo Accounts Receivable Prediction")
