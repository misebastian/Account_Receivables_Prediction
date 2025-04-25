# ──────────────────────────────────────────────────────────────
# App_AR_PRED.py  ·  v6  (25-Abr-2025)
# Ahora con filtros extra + nuevos KPIs + time-series + top clientes
# ──────────────────────────────────────────────────────────────
import warnings, io, base64, json
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")
st.set_page_config("AR Predictor", "📈", layout="wide")

# ───────────────────────────────────────── CONSTANTES
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")

# ───────────────────────────────────────── CSS
st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child {width: 270px;}
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────── LOADERS
@st.cache_data(show_spinner=False)
def load_data(fp:Path)->pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes":1,"No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1,"Paper":0})
    for c in ["InvoiceDate","DueDate"]:
        df[c]=pd.to_datetime(df[c])
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[
            df[c].dt.year, df[c].dt.month, df[c].dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_model(fp:Path):
    return joblib.load(fp)

raw_df = load_data(DATA_FILE)
model  = load_model(MODEL_FILE)
FEAT_COLS = model.feature_names_in_.tolist() if hasattr(model,"feature_names_in_") else model[-1].feature_names_in_.tolist()

# ───────────────────────────────────────── SIDEBAR  – Filtros
st.sidebar.header("Filtros")
min_d, max_d = raw_df["InvoiceDate"].min().date(), raw_df["InvoiceDate"].max().date()
f_date = st.sidebar.date_input("InvoiceDate rango", (min_d, max_d), min_value=min_d, max_value=max_d)
f_country = st.sidebar.multiselect("País (countryCode)",
                                   sorted(raw_df["countryCode"].unique().tolist()),
                                   default=sorted(raw_df["countryCode"].unique().tolist())[:5])
f_disputed_only = st.sidebar.checkbox("Solo facturas en disputa", False)

f_mask = (
    (raw_df["InvoiceDate"].between(pd.to_datetime(f_date[0]), pd.to_datetime(f_date[1]))) &
    (raw_df["countryCode"].isin(f_country))
)
if f_disputed_only:
    f_mask &= raw_df["Disputed"]==1

df = raw_df.loc[f_mask].copy()

# ─────────────────────────────────────────  KPIs
tot_inv = len(df)
dispute_pct = df["Disputed"].mean()*100
avg_amt_per_cust = df.groupby("customerID")["InvoiceAmount"].sum().mean()
aging = pd.cut(df["DaysLate"],
               bins=[-np.inf,0,30,60,np.inf],
               labels=["≤0d","1-30d","31-60d",">60d"]).value_counts().reindex(["≤0d","1-30d","31-60d",">60d"]).fillna(0)

k1,k2,k3,k4 = st.columns(4)
k1.metric("Facturas filtradas", f"{tot_inv:,}")
k2.metric("% en disputa", f"{dispute_pct:,.1f}%")
k3.metric("Importe medio / cliente", f"${avg_amt_per_cust:,.0f}")
k4.metric("Retrasos >60d", int(aging[">60d"]))

# ─────────────────────────────────────────  EDA  (variable seleccionada)
st.markdown("### 🔍 Análisis de variable")
sel = st.selectbox("Variable para explorar", df.columns, index=df.columns.get_loc("InvoiceAmount"))

if pd.api.types.is_numeric_dtype(df[sel]):
    colh, colb = st.columns(2)
    colh.plotly_chart(px.histogram(df, x=sel, nbins=40,
                                   color_discrete_sequence=["#2a9df4"]), use_container_width=True)
    colb.plotly_chart(px.box(df, y=sel, points="all",
                             color_discrete_sequence=["#e74c3c"]), use_container_width=True)
    st.dataframe(df[sel].describe().to_frame().T.round(2))
else:
    cnt = df[sel].value_counts().reset_index()
    st.plotly_chart(px.bar(cnt, x=sel, y="count", text_auto=True,
                           color_discrete_sequence=["#16a085"]), use_container_width=True)
    st.dataframe(cnt.rename(columns={sel:"valor","count":"conteo"}))

# Correlación dedicada con DaysLate
if pd.api.types.is_numeric_dtype(df[sel]):
    corr_val = df["DaysLate"].corr(df[sel])
    st.info(f"💡 **Correlación con DaysLate**: {corr_val:+.2f}")

# Matriz de correlación (selección múltiple)
with st.expander("Matriz de correlación (numéricas)"):
    num_cols = st.multiselect("Columnas numéricas", df.select_dtypes(include=[np.number]).columns,
                              default=["DaysLate","InvoiceAmount"])
    if len(num_cols)>=2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1,zmax=1,
                        height=400)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────  SERIE TEMPORAL
st.markdown("### 📈 Serie temporal")
ts_df = (df
         .groupby(pd.Grouper(key="InvoiceDate",freq="M"))
         .agg(dayslate_mean=("DaysLate","mean"),
              amt_sum=("InvoiceAmount","sum"))
         .reset_index())
fig_ts = go.Figure()
fig_ts.add_scatter(x=ts_df["InvoiceDate"], y=ts_df["dayslate_mean"],
                   mode="lines+markers", name="DaysLate medio")
fig_ts.add_bar(x=ts_df["InvoiceDate"], y=ts_df["amt_sum"],
               name="Importe total", yaxis="y2", opacity=0.4)
fig_ts.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                 title="Importe total"))
st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────  TOP CLIENTES
st.markdown("### 🏆 Top clientes por importe")
top_n = st.slider("N° clientes", 5, 20, 10, key="topn")
top = (df.groupby("customerID")["InvoiceAmount"].sum()
       .nlargest(top_n).reset_index())
st.plotly_chart(px.bar(top, x="InvoiceAmount", y="customerID",
                       orientation="h", text_auto=".2s",
                       color_discrete_sequence=["#f39c12"])
                .update_layout(yaxis=dict(categoryorder="total ascending")),
                use_container_width=True)

# ─────────────────────────────────────────  PREDICCIÓN
st.markdown("---")
st.header("🧮 Predicción de DaysLate")

with st.form("pred"):
    c1,c2 = st.columns(2)
    country  = c1.selectbox("countryCode", sorted(raw_df["countryCode"].unique()))
    inv_amt  = c1.number_input("InvoiceAmount", 0.0, 1e9, step=100.0, value=100.0)
    disputed = c1.selectbox("Disputed", ["No","Yes"])
    paper    = c1.selectbox("PaperlessBill", ["Paper","Electronic"])
    days_set = c1.number_input("DaysToSettle", 0, 365, 30)

    inv_date = c2.date_input("InvoiceDate", date(2013,9,1))
    due_date = c2.date_input("DueDate",    date(2013,10,1))

    subm = st.form_submit_button("Predecir")

if subm:
    row = {
        "countryCode": country,
        "InvoiceAmount": inv_amt,
        "Disputed": 1 if disputed=="Yes" else 0,
        "PaperlessBill": 1 if paper=="Electronic" else 0,
        "DaysToSettle": days_set,
        "InvoiceDate_year": inv_date.year,
        "InvoiceDate_month": inv_date.month,
        "InvoiceDate_day": inv_date.day,
        "DueDate_year": due_date.year,
        "DueDate_month": due_date.month,
        "DueDate_day": due_date.day,
    }
    X_new = pd.DataFrame([row])[FEAT_COLS]
    pred  = float(model.predict(X_new)[0])
    if pred<=0:
        st.success(f"✅ Pago estimado {abs(pred):.1f} días antes del vencimiento.")
    else:
        st.error(f"⏰ Retraso estimado de {pred:.1f} días.")

st.caption("© 2025 – Demo Accounts Receivable Prediction")

