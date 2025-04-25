# ──────────────────────────────────────────────────────────────
#  App_AR_PRED.py   ·   EDA + Predicción DaysLate (XGB Pipeline)
#  Repositorio:  ar_pipeline.pkl  |  WA_Fn-UseC_-Accounts-Receivable.xlsx
#  2025-04-25
# ──────────────────────────────────────────────────────────────
import warnings, base64, io
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ╭─────────────────────────── CONSTANTES ───────────────────────────╮
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")              # modelo + scaler
FEAT_COLS  = [                                    # columnas del entrenamiento
    "countryCode","InvoiceAmount","Disputed","PaperlessBill","DaysToSettle",
    "InvoiceDate_year","InvoiceDate_month","InvoiceDate_day",
    "DueDate_year","DueDate_month","DueDate_day"
]
# ╰───────────────────────────────────────────────────────────────────╯


# ──────────────────────── HELPERS ─────────────────────────
@st.cache_data(show_spinner=False)
def load_raw(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    # map binarios como en el entrenamiento
    df["Disputed"]      = df["Disputed"].map({"Yes":1, "No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1, "Paper":0})
    # fechas
    for c in ["InvoiceDate","DueDate"]:
        df[c] = pd.to_datetime(df[c])
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[
            df[c].dt.year, df[c].dt.month, df[c].dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_pipeline(fp: Path):
    return joblib.load(fp)

def make_kpi(df: pd.DataFrame):
    on_time_pct = (df["DaysLate"]<=0).mean()*100
    disputed_pct= (df["Disputed"]==1).mean()*100
    return on_time_pct, disputed_pct

# ──────────────────────── CARGA ───────────────────────────
raw_df = load_raw(DATA_FILE)
model   = load_pipeline(MODEL_FILE)
STORED_COLS = list(model.feature_names_in_)       # seguridad

# ─────────────────────────── UI GENERAL ────────────────────────────
st.set_page_config(page_title="AR Predictor", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    /* dark theme coordenado con Streamlit dark */
    .st-emotion-cache-1avcm0n {padding-top:0rem;}
    #MainMenu, footer {visibility:hidden;}
    section[data-testid="stSidebar"] > div:first-child {width:260px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────── 1. EDA ───────────────────────────────
st.title("📊 Exploratorio – Accounts Receivable")

k1,k2,k3 = st.columns(3)
on_time_pct, disputed_pct = make_kpi(raw_df)
k1.metric("Facturas",          f"{len(raw_df):,}")
k2.metric("Retraso medio",     f"{raw_df['DaysLate'].mean():.1f} días")
k3.metric("% a tiempo",        f"{on_time_pct:,.1f}%")

st.divider()

num_cols = raw_df.select_dtypes(include="number").columns.tolist()
sel = st.sidebar.selectbox("Variable numérica", num_cols, index=num_cols.index("InvoiceAmount"))

colH,colB = st.columns(2)
colH.plotly_chart(px.histogram(raw_df,x=sel,nbins=40,color_discrete_sequence=["#1f77b4"]),use_container_width=True)
colB.plotly_chart(px.box(raw_df,y=sel,color_discrete_sequence=["#d62728"]),use_container_width=True)

with st.expander("Matriz de correlación"):
    corr = raw_df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.columns,
                               colorscale="RdYlBu_r",zmin=-1,zmax=1))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ────────────────── 2. PREDICCIÓN INTERACTIVA ─────────────────────
st.header("🧮 Predicción de DaysLate")

with st.form("pred_form"):
    cL,cR = st.columns(2)
    with cL:
        cc     = st.selectbox("countryCode", sorted(raw_df["countryCode"].unique()))
        amt    = st.number_input("InvoiceAmount", 0.0, 1e9, float(raw_df["InvoiceAmount"].median()))
        disp   = st.selectbox("Disputed", ["No","Yes"])
        paper  = st.selectbox("PaperlessBill", ["Paper","Electronic"])
        dsettle= st.number_input("DaysToSettle", 0, 365, int(raw_df["DaysToSettle"].median()))
    with cR:
        inv_d  = st.date_input("InvoiceDate", value=date(2013,9,1))
        due_d  = st.date_input("DueDate",    value=date(2013,10,1))
    ok = st.form_submit_button("Predecir")

if ok:
    row = {
        "countryCode": cc,
        "InvoiceAmount": amt,
        "Disputed": 1 if disp=="Yes" else 0,
        "PaperlessBill": 1 if paper=="Electronic" else 0,
        "DaysToSettle": dsettle,
        "InvoiceDate_year": inv_d.year,
        "InvoiceDate_month": inv_d.month,
        "InvoiceDate_day": inv_d.day,
        "DueDate_year": due_d.year,
        "DueDate_month": due_d.month,
        "DueDate_day": due_d.day,
    }
    X_new = pd.DataFrame([row])

    # --- match columnas del modelo ---
    missing = [c for c in STORED_COLS if c not in X_new.columns]
    for m in missing: X_new[m] = 0
    X_new = X_new[STORED_COLS]

    pred = float(model.predict(X_new)[0])
    if pred <= 0:
        st.success(f"✅ Pago previsto **{abs(pred):.1f} días antes** del vencimiento.")
    else:
        st.error(f"🚨 Retraso estimado de **{pred:.1f} días**.")

st.caption("© 2025 – Demo Accounts Receivable Prediction")
