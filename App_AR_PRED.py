# App_AR_PRED.py  ·  Streamlit dashboard + predictor  (versión compacta)
# ────────────────────────────────────────────────────────────
import json, joblib, warnings, itertools
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config("AR Dashboard", "📈", layout="wide")

# ────────────────────────────────────────────────────────────
# 1 ▸ Rutas de artefactos
# ────────────────────────────────────────────────────────────
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")

# ────────────────────────────────────────────────────────────
# 2 ▸ Func. cacheadas
# ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Disputed"]      = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})
    for c in ["InvoiceDate", "DueDate"]:
        df[c] = pd.to_datetime(df[c])
        df[[f"{c}_year", f"{c}_month", f"{c}_day"]] = np.c_[
            df[c].dt.year, df[c].dt.month, df[c].dt.day
        ]
    # aging buckets
    cuts = [-9999, 0, 30, 60, 90, np.inf]
    labels = ["OnTime", "1-30", "31-60", "61-90", "90+"]
    df["AgingBucket"] = pd.cut(df["DaysLate"], bins=cuts, labels=labels)
    return df

@st.cache_resource(show_spinner=False)
def load_model(fp: Path):
    return joblib.load(fp)

raw_df = load_data(DATA_FILE)
model  = load_model(MODEL_FILE)
FEAT_COLS = model.feature_names_in_

# ────────────────────────────────────────────────────────────
# 3 ▸ SIDEBAR · filtros y variable(s) de interés
# ────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filtros")
min_d, max_d = raw_df.InvoiceDate.min(), raw_df.InvoiceDate.max()
date_rng = st.sidebar.date_input(
    "Rango de InvoiceDate", (min_d, max_d), min_value=min_d, max_value=max_d
)
sel_ctry = st.sidebar.multiselect(
    "País / countryCode", sorted(raw_df.countryCode.unique().tolist()),
    default=[]
)
sel_disp = st.sidebar.multiselect(
    "Disputed", ["No dispute", "Disputed"], default=[]
)

# variable(s) para análisis
num_cols = raw_df.select_dtypes("number").columns.drop("DaysLate")
var_list = st.sidebar.multiselect(
    "Variables numéricas a analizar ⬇", num_cols, default=["InvoiceAmount"]
)

# aplica filtros
df = raw_df.copy()
df = df[(df.InvoiceDate >= pd.Timestamp(date_rng[0])) &
        (df.InvoiceDate <= pd.Timestamp(date_rng[1]))]
if sel_ctry:
    df = df[df.countryCode.isin(sel_ctry)]
if sel_disp:
    df = df[df.Disputed.isin([1 if s == "Disputed" else 0 for s in sel_disp])]

# ────────────────────────────────────────────────────────────
# 4 ▸ KPIs
# ────────────────────────────────────────────────────────────
kpiA, kpiB, kpiC, kpiD, kpiE = st.columns(5)
kpiA.metric("Facturas", f"{len(df):,}")
pct_disp = (df.Disputed.mean()*100) if len(df) else 0
kpiB.metric("% Disputas", f"{pct_disp:,.1f}%")
kpiC.metric("Importe medio", f"{df.InvoiceAmount.mean():,.2f}")
kpiD.metric("DaysLate medio", f"{df.DaysLate.mean():.2f}")
kpiE.metric("Importe medio / cliente",
            f"{df.groupby('customerID').InvoiceAmount.mean().mean():,.2f}")

st.markdown("---")

# ────────────────────────────────────────────────────────────
# 5 ▸ Distribuciones rápidas
# ────────────────────────────────────────────────────────────
with st.expander("📊 Histogramas & Boxplot", expanded=False):
    for v in var_list:
        h = px.histogram(df, x=v, marginal="box", nbins=40,
                         color_discrete_sequence=["#2A9DF4"])
        st.plotly_chart(h, use_container_width=True)

# ────────────────────────────────────────────────────────────
# 6 ▸ Aging buckets · Heatmap + Área apilada (tiempo)
# ────────────────────────────────────────────────────────────
st.subheader("⏱️ Aging buckets")

col_age1, col_age2 = st.columns((2,3))

age_counts = df.AgingBucket.value_counts().reindex(
    ["OnTime", "1-30", "31-60", "61-90", "90+"]).fillna(0)

col_age1.table(age_counts.to_frame("Facturas"))

# heatmap por país
pivot_age = (df.pivot_table(index="countryCode",
                            columns="AgingBucket",
                            values="InvoiceAmount",
                            aggfunc="sum",
                            fill_value=0)
               .reindex(columns=["OnTime","1-30","31-60","61-90","90+"]))
hm = go.Figure(go.Heatmap(
    z=pivot_age.values, x=pivot_age.columns, y=pivot_age.index,
    colorscale="Blues"))
hm.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0))
col_age2.plotly_chart(hm, use_container_width=True)

# series temporal apilada
st.subheader("📈 Evolución mensual de Aging (%)")
monthly = (
    df.groupby([pd.Grouper(key="InvoiceDate", freq="M"), "AgingBucket"])
      .size().reset_index(name="Count")
)

monthly["Pct"] = monthly.groupby("InvoiceDate").Count.transform(
    lambda x: x / x.sum() * 100
)
fig_area = px.area(monthly, x="InvoiceDate", y="Pct", color="AgingBucket",
                   category_orders={"AgingBucket":["OnTime","1-30","31-60",
                                                   "61-90","90+"]})
fig_area.update_layout(legend_title=None, yaxis_title="%")
st.plotly_chart(fig_area, use_container_width=True)

# ────────────────────────────────────────────────────────────
# 7 ▸ Top-10 clientes
# ────────────────────────────────────────────────────────────
st.subheader("🏆 Top-10 clientes (importe facturado)")

g_sum = (df.groupby("customerID")
           .InvoiceAmount.sum()
           .nlargest(10)
           .sort_values())
fig_top = px.bar(x=g_sum.values, y=g_sum.index, orientation="h",
                 labels={"x":"Importe","y":"customerID"},
                 color_discrete_sequence=["#16A085"])
st.plotly_chart(fig_top, use_container_width=True)

# ────────────────────────────────────────────────────────────
# 8 ▸ Correlación (multiselección)
# ────────────────────────────────────────────────────────────
st.subheader("🔗 Correlación")
corr_vars = st.multiselect(
    "Variables para matriz de correlación",
    options=df.select_dtypes("number").columns.tolist(),
    default=list(var_list)+["DaysLate"]
)
if len(corr_vars) >= 2:
    corr = df[corr_vars].corr()
    heat = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                        colorscale="RdBu", zmin=-1, zmax=1))
    heat.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(heat, use_container_width=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────
# 9 ▸ Predicción interactiva
# ────────────────────────────────────────────────────────────
st.header("🧮 Predicción de DaysLate (XGB)")

with st.form("pred"):
    c1, c2, c3 = st.columns(3)
    with c1:
        cc   = st.selectbox("countryCode", sorted(raw_df.countryCode.unique()))
        amt  = st.number_input("InvoiceAmount", 0.0, 99999.0, 50.0, step=1.0)
        disp = st.selectbox("Disputed", ["No", "Yes"])
    with c2:
        papl = st.selectbox("PaperlessBill", ["Paper","Electronic"])
        dts  = st.number_input("DaysToSettle", 0, 120, 30, step=1)
    with c3:
        iday = st.date_input("InvoiceDate", raw_df.InvoiceDate.min())
        dday = st.date_input("DueDate",     raw_df.DueDate.min())
    submit = st.form_submit_button("Predecir")

if submit:
    row = {
        "countryCode": cc,
        "InvoiceAmount": amt,
        "Disputed": 1 if disp=="Yes" else 0,
        "PaperlessBill": 1 if papl=="Electronic" else 0,
        "DaysToSettle": dts,
        "InvoiceDate_year": iday.year,
        "InvoiceDate_month": iday.month,
        "InvoiceDate_day": iday.day,
        "DueDate_year": dday.year,
        "DueDate_month": dday.month,
        "DueDate_day": dday.day,
    }
    Xnew = pd.DataFrame([row])[FEAT_COLS]
    pred = model.predict(Xnew)[0]
    if pred < 0:
        st.success(f"✅ Se espera pago {abs(pred):.1f} días ANTES del vencimiento.")
    else:
        st.error(f"🚨 Retraso estimado: {pred:.1f} días.")

# ────────────────────────────────────────────────────────────
# 10 ▸ Conclusiones clave
# ────────────────────────────────────────────────────────────
with st.expander("📌 Conclusiones rápidas"):
    st.markdown("""
* **Concentración de riesgo** → Un ~{:.1f}% de la cartera > 60 días se concentra en los 10 clientes principales.  
* **{}% de las facturas están en disputa**; estas presentan un *DaysLate* medio **{:.1f}×** mayor que las no disputadas.  
* El **aging bucket 90+** ha {} en los últimos 6 meses.  
""".format(
        (df[df.AgingBucket.isin(["61-90","90+"])].InvoiceAmount.sum() /
         df.InvoiceAmount.sum()*100) if len(df) else 0,
        f"{pct_disp:.1f}",
        (df[df.Disputed==1].DaysLate.mean() / df[df.Disputed==0].DaysLate.mean())
          if df.Disputed.eq(1).any() and df.Disputed.eq(0).any() else 1,
        "crecido" if monthly.query("AgingBucket=='90+'").Pct.tail(1).values[0] >
                     monthly.query("AgingBucket=='90+'").Pct.head(1).values[0]
                 else "disminuido"
    ))
st.caption("© 2025 – AR Predictor · dashboard extendido")

