# ──────────────────────────────────────────────────────────────
#  AR Predictor · Streamlit v6 (compact-dark) – 25 Abr 2025
# ──────────────────────────────────────────────────────────────
import warnings, json, base64, io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import iqr
import joblib

st.set_page_config("AR Predictor", "😎", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")

#───────────  utilidades  ──────────────────────────────────────
DATA_FILE  = Path(__file__).with_name("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path(__file__).with_name("ar_pipeline.pkl")

FEAT_COLS = ["countryCode","InvoiceAmount","Disputed","PaperlessBill","DaysToSettle",
             "InvoiceDate_year","InvoiceDate_month","InvoiceDate_day",
             "DueDate_year","DueDate_month","DueDate_day"]

def format_big(x): return f"{x:,.0f}".replace(",", " ")

#───────────  carga datos + modelo  ────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(fp)->pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes":1,"No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1,"Paper":0})
    for c in ["InvoiceDate","DueDate"]:
        dt = pd.to_datetime(df[c])
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[dt.dt.year,dt.dt.month,dt.dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_model(fp):
    return joblib.load(fp)

raw_df       = load_data(DATA_FILE)
model        = load_model(MODEL_FILE)

#───────────  KPIs (encabezado)  ───────────────────────────────
st.markdown("## 📊 *Exploratorio* – Accounts Receivable")

tot_inv  = len(raw_df)
unique_c = raw_df.customerID.nunique()
late_pct = (raw_df.DaysLate>0).mean()*100
avg_dl   = raw_df.DaysLate.mean()

k1,k2,k3,k4 = st.columns(4)
k1.metric("Facturas", format_big(tot_inv))
k2.metric("Clientes únicos", format_big(unique_c))
k3.metric("% Con retraso", f"{late_pct:.1f}%")
k4.metric("DaysLate medio", f"{avg_dl:.1f}")

st.divider()

#───────────  selector de variable + gráficas  ─────────────────
all_cols = raw_df.columns.tolist()
sel = st.sidebar.selectbox("Variable para explorar", all_cols, index=all_cols.index("InvoiceAmount"))
st.subheader(f"🔎 Análisis de **{sel}**")

def draw_conclusion(txt, emoji="💡"):
    st.markdown(f"{emoji} {txt}")

if pd.api.types.is_numeric_dtype(raw_df[sel]):           # NUMÉRICA
    colL,colR = st.columns(2)
    colL.plotly_chart(px.histogram(raw_df,x=sel,nbins=40,
                                   color_discrete_sequence=["#2a9df4"],
                                   height=260), use_container_width=True)
    colR.plotly_chart(px.box(raw_df,y=sel,
                             color_discrete_sequence=["#e74c3c"],
                             height=260), use_container_width=True)

    desc = pd.Series({
        "suma":   raw_df[sel].sum(),
        "media":  raw_df[sel].mean(),
        "mediana":raw_df[sel].median(),
        "std":    raw_df[sel].std(),
        "IQR":    iqr(raw_df[sel])
    }).to_frame("valor").round(3)
    st.dataframe(desc)

    # correlación sencilla con DaysLate
    corr = raw_df[[sel,"DaysLate"]].corr().iloc[0,1]
    draw_conclusion(f"Correlación con **DaysLate**: {corr:+.2f}")

else:                                                    # CATEGÓRICA
    vc = (raw_df[sel].astype(str)
                     .value_counts()
                     .rename_axis(sel)
                     .reset_index(name="count")
                     .head(30))
    st.plotly_chart(px.bar(vc,x=sel,y="count",
                           color_discrete_sequence=["#27ae60"],
                           height=300), use_container_width=True)

    agg = (raw_df.groupby(sel)
                   .agg(facturas=("invoiceNumber","count"),
                        suma_monto=("InvoiceAmount","sum"),
                        retraso_medio=("DaysLate","mean"))
                   .sort_values("facturas",ascending=False)
                   .head(10).round(2))
    st.dataframe(agg)

    worst = agg.retraso_medio.nlargest(3).index.astype(str)
    draw_conclusion(f"Mayores retrasos medios en: {', '.join(worst)}.")

#───────────  matriz de correlación pequeña  ──────────────────
with st.expander("Matriz de correlación (numéricas)", expanded=False):
    num = raw_df.select_dtypes(float)
    corr = num.corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True,
                              color_continuous_scale="RdBu_r",
                              aspect="auto",
                              height=500), use_container_width=True)

st.divider()

#───────────  PREDICCIÓN interactiva  ─────────────────────────
st.markdown("## 🧮 Predicción de **DaysLate**")

with st.form("pred"):
    l,r = st.columns(2)
    cc      = l.selectbox("countryCode", sorted(raw_df.countryCode.unique()))
    inv_amt = l.number_input("InvoiceAmount", 0.0, 1e7, 50.0, step=10.0)
    disputed= l.selectbox("Disputed", ["No","Yes"])
    paper   = l.selectbox("PaperlessBill", ["Paper","Electronic"])
    days_set= l.number_input("DaysToSettle", 0, 365, 30, step=1)

    inv_date= r.date_input("InvoiceDate", pd.Timestamp("2013-09-01"))
    due_date= r.date_input("DueDate",   pd.Timestamp("2013-10-01"))

    submit  = st.form_submit_button("Predecir")

if submit:
    row = {
        "countryCode":      float(cc),
        "InvoiceAmount":    float(inv_amt),
        "Disputed":         1.0 if disputed=="Yes" else 0.0,
        "PaperlessBill":    1.0 if paper=="Electronic" else 0.0,
        "DaysToSettle":     float(days_set),
        "InvoiceDate_year": inv_date.year,
        "InvoiceDate_month":inv_date.month,
        "InvoiceDate_day":  inv_date.day,
        "DueDate_year":     due_date.year,
        "DueDate_month":    due_date.month,
        "DueDate_day":      due_date.day
    }
    X_new = pd.DataFrame([row])[FEAT_COLS]
    pred  = float(model.predict(X_new)[0])

    if pred <= 0:
        st.success(f"🎉 Pago estimado **{abs(pred):.1f} días antes** del vencimiento.")
    else:
        st.error(f"⏰ Retraso estimado de **{pred:.1f} días**.")

st.caption("© 2025 – Demo Accounts Receivable Prediction")

