# ─────────────────────────────────────────────────────────────
# App_AR_PRED.py · Streamlit v5 (selector global + análisis)
# ─────────────────────────────────────────────────────────────
import warnings, base64, io, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from scipy.stats import iqr, skew, kurtosis

warnings.filterwarnings("ignore")
st.set_page_config("AR Predictor", "📈", "wide")

# ── Archivos requeridos
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")

# ╭───────────────────────────────────────╮
# │ 1 · Carga de datos y pipeline         │
# ╰───────────────────────────────────────╯
@st.cache_data(show_spinner=False)
def load_raw(fp):
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
def load_pipe(fp):
    return joblib.load(fp)

if not (DATA_FILE.exists() and MODEL_FILE.exists()):
    st.error("🚫  Faltan archivos en el repositorio.")
    st.stop()

df = load_raw(DATA_FILE)
pipe = load_pipe(MODEL_FILE)

FEATS = ["countryCode","InvoiceAmount","Disputed","PaperlessBill","DaysToSettle",
         "InvoiceDate_year","InvoiceDate_month","InvoiceDate_day",
         "DueDate_year","DueDate_month","DueDate_day"]

# ╭───────────────────────────────────────╮
# │ 2 · KPI compactos                     │
# ╰───────────────────────────────────────╯
st.markdown("""
<style>
.small-card {padding:0.2rem 0.4rem !important}
.metric .metric-label{font-size:0.75rem;color:gray}
.metric .metric-value{font-size:1.05rem}
section[data-testid="stSidebar"]>div:first-child {width:240px}
#MainMenu, footer {visibility:hidden}
</style>""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.metric("Facturas", f"{len(df):,}")
k2.metric("Clientes", f"{df.customerID.nunique():,}")
k3.metric("Retraso medio", f"{df.DaysLate.mean():.1f} d")
k4.metric("% puntuales", f"{(df.DaysLate<=0).mean()*100:.1f}%")

# ╭───────────────────────────────────────╮
# │ 3 · Selector global y visualización   │
# ╰───────────────────────────────────────╯
all_cols = df.columns.tolist()
sel = st.sidebar.selectbox("Variable para explorar", all_cols, index=all_cols.index("InvoiceAmount"))

st.subheader(f"🔎 Análisis de **{sel}**", divider="gray")

if pd.api.types.is_numeric_dtype(df[sel]):
    colA,colB = st.columns(2)
    colA.plotly_chart(
        px.histogram(df, x=sel, nbins=40, height=250,
                     color_discrete_sequence=["#2980b9"]), use_container_width=True)
    colB.plotly_chart(
        px.box(df, y=sel, height=250,
               color_discrete_sequence=["#c0392b"]), use_container_width=True)
else:
    st.plotly_chart(
        px.bar(df[sel].value_counts().reset_index(),
               x="index", y=sel, height=300,
               labels={"index":sel,"y":"Frecuencia"},
               color_discrete_sequence=["#27ae60"]),
        use_container_width=True)

# ── Estadística descriptiva extendida
with st.expander("📊  Estadísticos y ejemplos"):
    desc = df[sel].describe(include="all").to_frame().T
    if pd.api.types.is_numeric_dtype(df[sel]):
        desc["median"]   = df[sel].median()
        desc["IQR"]      = iqr(df[sel])
        desc["cv"]       = df[sel].std()/df[sel].mean()
    st.dataframe(desc.round(3))

    c_top, c_bot = st.columns(2)
    c_top.markdown("**Top-5**")
    c_top.dataframe(df.nlargest(5, sel)[[sel]].reset_index(drop=True))
    c_bot.markdown("**Bottom-5**")
    c_bot.dataframe(df.nsmallest(5, sel)[[sel]].reset_index(drop=True))

# ╭───────────────────────────────────────╮
# │ 4 · Vista especial de DaysLate        │
# ╰───────────────────────────────────────╯
with st.expander("⏱  Análisis de la variable objetivo (DaysLate)", expanded=False):
    dl = df["DaysLate"]
    h = px.histogram(dl, nbins=40, marginal="rug",
                     title="Distribución de DaysLate",
                     color_discrete_sequence=["#8e44ad"])
    st.plotly_chart(h, use_container_width=True)

    stats = pd.Series({
        "mean": dl.mean(), "median": dl.median(),
        "IQR": iqr(dl), "skew": skew(dl), "kurtosis": kurtosis(dl)
    }).round(3)
    st.write("**Resúmen numérico**")
    st.dataframe(stats.to_frame("value"))

    worst = (df.groupby("customerID")["DaysLate"]
               .mean().sort_values(ascending=False).head(10).round(1))
    st.write("**Clientes con mayor retraso medio:**")
    st.table(worst)

# ╭───────────────────────────────────────╮
# │ 5 · Formulario de predicción          │
# ╰───────────────────────────────────────╯
st.header("🎯 Predicción de DaysLate", divider="gray")
with st.form("pred"):
    c1,c2 = st.columns(2)
    # Columna izquierda
    cc   = c1.selectbox("countryCode", sorted(df.countryCode.unique()))
    amt  = c1.number_input("InvoiceAmount", 0.0, 1e6, 50.0, step=10.0)
    disp = c1.selectbox("Disputed", ["No","Yes"])
    pbl  = c1.selectbox("PaperlessBill", ["Paper","Electronic"])
    dts  = c1.number_input("DaysToSettle", 0, 120, 30)

    # Columna derecha
    inv  = c2.date_input("InvoiceDate", dt.date(2013, 9, 1))
    due  = c2.date_input("DueDate",    dt.date(2013,10, 1))

    ok = st.form_submit_button("Predecir")

if ok:
    row = {
        "countryCode": cc, "InvoiceAmount": amt,
        "Disputed": 1 if disp=="Yes" else 0,
        "PaperlessBill": 1 if pbl=="Electronic" else 0,
        "DaysToSettle": dts,
        "InvoiceDate_year":  inv.year, "InvoiceDate_month": inv.month, "InvoiceDate_day": inv.day,
        "DueDate_year":      due.year, "DueDate_month":    due.month, "DueDate_day":    due.day
    }
    Xnew = pd.DataFrame([row])[FEATS]
    pred = float(pipe.predict(Xnew)[0])
    if pred <= 0:
        st.success(f"✅ Pago estimado **{abs(pred):.1f} d antes** del vencimiento.")
    else:
        st.error(f"🚨 Retraso estimado de **{pred:.1f} días**.")

st.caption("© 2025 – Demo Accounts Receivable Prediction | v5")
