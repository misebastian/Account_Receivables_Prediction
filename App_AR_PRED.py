# ──────────────────────────────────────────────────────────────
#  AR Predictor · Streamlit v7 – 25 Abr 2025
#    · selector de variables en matriz de correlación
# ──────────────────────────────────────────────────────────────
import warnings, base64, io
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import iqr
import joblib

st.set_page_config("AR Predictor", "😎", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")

DATA_FILE  = Path(__file__).with_name("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path(__file__).with_name("ar_pipeline.pkl")

FEAT_COLS = ["countryCode","InvoiceAmount","Disputed","PaperlessBill","DaysToSettle",
             "InvoiceDate_year","InvoiceDate_month","InvoiceDate_day",
             "DueDate_year","DueDate_month","DueDate_day"]

#────────────  helpers  ────────────────────────────────────────
def load_data(fp):
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes":1,"No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1,"Paper":0})
    for c in ["InvoiceDate","DueDate"]:
        dt = pd.to_datetime(df[c])
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[dt.dt.year,dt.dt.month,dt.dt.day]
    return df

@st.cache_data(show_spinner=False)   # datos se cachean
def get_df(): return load_data(DATA_FILE)

@st.cache_resource(show_spinner=False)
def get_model(): return joblib.load(MODEL_FILE)

#────────────  DATA & MODEL  ───────────────────────────────────
raw_df = get_df()
model  = get_model()

#────────────  KPIs  ───────────────────────────────────────────
tot_inv  = len(raw_df)
unique_c = raw_df.customerID.nunique()
late_pct = (raw_df.DaysLate>0).mean()*100
avg_dl   = raw_df.DaysLate.mean()

st.markdown("## 📊 *Exploratorio* – Accounts Receivable")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Facturas", f"{tot_inv:,}")
k2.metric("Clientes únicos", f"{unique_c:,}")
k3.metric("% Con retraso", f"{late_pct:.1f}%")
k4.metric("DaysLate medio", f"{avg_dl:.1f}")
st.divider()

#────────────  Análisis variable  ──────────────────────────────
all_cols = raw_df.columns.tolist()
sel = st.sidebar.selectbox("Variable para explorar", all_cols, index=all_cols.index("InvoiceAmount"))
st.subheader(f"🔎 Análisis de **{sel}**")

if pd.api.types.is_numeric_dtype(raw_df[sel]):
    a,b = st.columns(2)
    a.plotly_chart(px.histogram(raw_df,x=sel,nbins=40,color_discrete_sequence=["#3498db"],height=260),
                   use_container_width=True)
    b.plotly_chart(px.box(raw_df,y=sel,color_discrete_sequence=["#e74c3c"],height=260),
                   use_container_width=True)
    stats = pd.Series({
        "suma":raw_df[sel].sum(),"media":raw_df[sel].mean(),
        "mediana":raw_df[sel].median(),"std":raw_df[sel].std(),
        "IQR":iqr(raw_df[sel])}).round(2)
    st.dataframe(stats.to_frame("valor"))
    corr = raw_df[[sel,"DaysLate"]].corr().iloc[0,1]
    st.markdown(f"💡 **Correlación con DaysLate**: {corr:+.2f}")
else:
    vc = (raw_df[sel].astype(str).value_counts().head(30)
           .rename_axis(sel).reset_index(name="count"))
    st.plotly_chart(px.bar(vc,x=sel,y="count",color_discrete_sequence=["#27ae60"]),
                    use_container_width=True)
    agg = (raw_df.groupby(sel)
           .agg(facturas=("invoiceNumber","count"),
                suma_monto=("InvoiceAmount","sum"),
                retraso_medio=("DaysLate","mean"))
           .sort_values("facturas",ascending=False).head(10).round(2))
    st.dataframe(agg)
    st.markdown("💡 Retraso medio máximo en: **{}**"
                .format(", ".join(agg.retraso_medio.nlargest(3).index.astype(str))))

#────────────  MATRIZ DE CORRELACIÓN PERSONALIZABLE  ──────────
with st.expander("Matriz de correlación (personalizable)"):
    num_cols = raw_df.select_dtypes(include="number").columns.tolist()
    sel_vars = st.multiselect("Selecciona variables numéricas (mín. 2)", num_cols,
                              default=num_cols if len(num_cols)<=8 else num_cols[:8])
    if len(sel_vars)>=2:
        corr = raw_df[sel_vars].corr().round(2)
        st.plotly_chart(px.imshow(corr,text_auto=True,color_continuous_scale="RdBu_r",
                                  aspect="auto",height=70+40*len(sel_vars)),
                        use_container_width=True)
    else:
        st.info("Selecciona al menos dos variables.")

st.divider()

#────────────  PREDICCIÓN  ─────────────────────────────────────
st.markdown("## 🧮 Predicción de **DaysLate**")
with st.form("pred"):
    l,r = st.columns(2)
    cc      = l.selectbox("countryCode", sorted(raw_df.countryCode.unique()))
    amt     = l.number_input("InvoiceAmount", 0.0, 1e7, 50.0, step=10.0)
    dispt   = l.selectbox("Disputed", ["No","Yes"])
    paper   = l.selectbox("PaperlessBill", ["Paper","Electronic"])
    dsettle = l.number_input("DaysToSettle", 0, 365, 30)
    inv_d   = r.date_input("InvoiceDate", pd.Timestamp("2013-09-01"))
    due_d   = r.date_input("DueDate",    pd.Timestamp("2013-10-01"))
    btn     = st.form_submit_button("Predecir")

if btn:
    row = {
        "countryCode":float(cc),"InvoiceAmount":float(amt),
        "Disputed":1.0 if dispt=="Yes" else 0.0,
        "PaperlessBill":1.0 if paper=="Electronic" else 0.0,
        "DaysToSettle":float(dsettle),
        "InvoiceDate_year":inv_d.year,"InvoiceDate_month":inv_d.month,"InvoiceDate_day":inv_d.day,
        "DueDate_year":due_d.year,"DueDate_month":due_d.month,"DueDate_day":due_d.day}
    pred = float(model.predict(pd.DataFrame([row])[FEAT_COLS])[0])
    st.success(f"⏱ **{pred:+.1f} días** (positivo → retraso, negativo → anticipado)")

st.caption("© 2025 – Demo AR Predictor v7")
