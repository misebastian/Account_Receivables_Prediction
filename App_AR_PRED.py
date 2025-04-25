# ────────────────────────────────────────────────────────────────
#  App_AR_PRED.py  ·  EDA + predicción DaysLate  (v5 compacto)
#  Repo:  https://github.com/miusuario/Account_Receivables_Prediction
# ────────────────────────────────────────────────────────────────
import warnings, json, base64, io
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib, streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AR Predictor", layout="wide", page_icon="📈")
warnings.filterwarnings("ignore")

# ---------- 1 · helpers -------------------------------------------------------
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")

@st.cache_data(show_spinner=False)
def load_data(fp: Path)->pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes":1,"No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1,"Paper":0})
    for c in ["InvoiceDate","DueDate"]:
        df[c]=pd.to_datetime(df[c], errors='coerce')
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[
            df[c].dt.year, df[c].dt.month, df[c].dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_model(fp: Path):
    return joblib.load(fp)

# ---------- 2 · Carga ---------------------------------------------------------
if not DATA_FILE.exists() or not MODEL_FILE.exists():
    st.error("❌ Faltan archivos de datos o modelo en el repo.")
    st.stop()

raw_df         = load_data(DATA_FILE)
model          = load_model(MODEL_FILE)
FEATURE_COLS   = model.feature_names_in_
NUM_COLS       = raw_df.select_dtypes("number").columns.tolist()
CAT_COLS       = [c for c in raw_df.columns if c not in NUM_COLS]

# ---------- 3 · KPI header ----------------------------------------------------
k1,k2,k3,k4 = st.columns(4)
k1.metric("🧾 Facturas",     f"{len(raw_df):,}")
k2.metric("👥 Clientes",     f"{raw_df['customerID'].nunique():,}")
k3.metric("💵 Total importe",f"${raw_df['InvoiceAmount'].sum():,.0f}")
k4.metric("⌛ Retraso medio", f"{raw_df['DaysLate'].mean():.1f} días")

st.markdown("## 🔎 EDA interactivo")

# ---------- 4 · Selector variable + datos previos -----------------------------
sel = st.sidebar.selectbox("Variable para explorar", raw_df.columns, index=raw_df.columns.get_loc("InvoiceAmount"))

df = raw_df  # alias corto

# ―― NUMÉRICA  ---------------------------------------------------------------
if sel in NUM_COLS:
    colL,colR = st.columns(2)
    hist = px.histogram(df, x=sel, nbins=40, color_discrete_sequence=["#1f77b4"])
    box  = px.box(df, y=sel, color_discrete_sequence=["#e74c3c"])
    colL.plotly_chart(hist, use_container_width=True)
    colR.plotly_chart(box,  use_container_width=True)

    st.dataframe(df[[sel]].describe().T.round(2))
# ―― CATEGÓRICA  -------------------------------------------------------------
else:
    vc = df[sel].value_counts().nlargest(20)  # top-20 p/ no petar
    fig_cat = px.bar(vc, text_auto=True, orientation="v",
                     color_discrete_sequence=["#16a085"])
    st.plotly_chart(fig_cat, use_container_width=True)
    st.dataframe(vc.to_frame("conteo"))

# ---------- 5 · Matriz de correlación multi-select ----------------------------
with st.expander("🔗 Matriz de correlación (numéricas)"):
    mult = st.multiselect("Variables a incluir",
                          NUM_COLS,
                          default=[c for c in NUM_COLS if c!=sel][:10])
    corr_df = df[[sel]+mult].corr()
    fig_corr = px.imshow(corr_df,
                         color_continuous_scale="RdBu_r",
                         zmin=-1,zmax=1,
                         text_auto=".2f", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ------- Conclusiones auto ------------
    st.markdown("### 📌 Conclusiones automáticas")
    target_corr = corr_df["DaysLate"].drop("DaysLate", errors="ignore")
    if not target_corr.empty:
        top_pos = target_corr.sort_values(ascending=False).head(3)
        top_neg = target_corr.sort_values().head(3)
        st.markdown("**🔹 Variables más correlacionadas (+):** "
                    + ", ".join(f"`{v}` ({c:.2f})" for v,c in top_pos.items()))
        st.markdown("**🔹 Variables más correlacionadas (−):** "
                    + ", ".join(f"`{v}` ({c:.2f})" for v,c in top_neg.items()))
    else:
        st.info("`DaysLate` no está en la matriz 🤷")

# ---------- 6 · Vista previa --------------------------------------------------
with st.expander("🔍 Ver primeras filas"):
    st.dataframe(df.head(8))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
#  7 · Predicciones
# ──────────────────────────────────────────────────────────────────────────────
st.header("🎯 Predicción de DaysLate")

with st.form("pred"):
    c1,c2 = st.columns(2)
    country   = c1.selectbox("countryCode", sorted(df["countryCode"].unique()))
    invdate   = c2.date_input("InvoiceDate", date(2013,9,1))
    invamt    = c1.number_input("InvoiceAmount", 0.0, 1e7, 100.0, 1.0)
    duedate   = c2.date_input("DueDate", date(2013,10,1))
    disputed  = c1.selectbox("Disputed", ["No","Yes"])
    paperless = c2.selectbox("PaperlessBill", ["Paper","Electronic"])
    dsettle   = c1.number_input("DaysToSettle", 0, 120, 30, 1)
    submitted = st.form_submit_button("Predecir")

if submitted:
    row = dict(
        countryCode   = float(country),
        InvoiceAmount = float(invamt),
        Disputed      = 1.0 if disputed=="Yes" else 0.0,
        PaperlessBill = 1.0 if paperless=="Electronic" else 0.0,
        DaysToSettle  = float(dsettle),
        InvoiceDate_year  = invdate.year,  InvoiceDate_month = invdate.month,
        InvoiceDate_day   = invdate.day,   DueDate_year      = duedate.year,
        DueDate_month     = duedate.month, DueDate_day       = duedate.day,
    )
    X_new = pd.DataFrame([row])[FEATURE_COLS]
    try:
        pred = float(model.predict(X_new)[0])
        if pred <= 0:
            st.success(f"✅ Pago estimado **{abs(pred):.1f} días antes** del vencimiento.")
        else:
            st.error(f"⏰ Retraso estimado de **{pred:.1f} días**.")
    except Exception as e:
        st.error("Error al predecir – revisa las variables.")
        st.exception(e)

st.caption("© 2025 – Demo Accounts Receivable Prediction")

