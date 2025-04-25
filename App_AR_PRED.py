# app.py ────────────────────────────────────────────────────────────
# Streamlit • EDA + predicción DaysLate con modelo cargado
# Requiere en requirements.txt:
#   streamlit>=1.32   pandas  numpy  plotly  joblib  openpyxl
# ────────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, joblib, json
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import date

# ──────────────── CONFIG ──────────────────────────────────────────
st.set_page_config("AR Predictor", "📊", layout="wide")
st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child {width: 240px;}
#MainMenu, footer {visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ──────────────── 1) DATA Y MODELO ───────────────────────────────
DATA_FILE  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_FILE = Path("ar_pipeline.pkl")
META_FILE  = Path("ar_meta.json")

@st.cache_data(show_spinner=False)
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    # mapeos categóricos
    df["Disputed"]      = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})
    # descomposición de fechas clave
    for c in ["InvoiceDate", "DueDate"]:
        dt = pd.to_datetime(df[c])
        df[[f"{c}_year", f"{c}_month", f"{c}_day"]] = np.c_[dt.dt.year,
                                                            dt.dt.month,
                                                            dt.dt.day]
    return df

@st.cache_resource(show_spinner=False)
def load_model_and_meta(model_fp: Path, meta_fp: Path):
    model = joblib.load(model_fp)
    with open(meta_fp, "r", encoding="utf-8") as f:
        meta  = json.load(f)
    feats = meta.get("features")        # lista de columnas usadas por el modelo
    return model, meta, feats

raw_df                  = load_data(DATA_FILE)
model, meta, FEAT_COLS  = load_model_and_meta(MODEL_FILE, META_FILE)

# si meta no tenía la lista, la deducimos quitando la target
if FEAT_COLS is None:
    FEAT_COLS = [c for c in raw_df.columns if c not in ["DaysLate"]]

# ──────────────── 2) CABECERA KPI ────────────────────────────────
st.title("📊 Exploratorio de Cuentas por Cobrar")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Facturas",       f"{len(raw_df):,}")
k2.metric("Prom. DaysLate", f"{raw_df['DaysLate'].mean():.2f}")
k3.metric("% A tiempo",     f"{(raw_df['DaysLate']<=0).mean()*100:.1f} %")
k4.metric("% > 5 d tarde",  f"{(raw_df['DaysLate']>5).mean()*100:.1f} %")

st.divider()

# ──────────────── 3) VARIABLE EXPLORER ───────────────────────────
num_vars = sorted([c for c, d in zip(raw_df.columns, raw_df.dtypes)
                   if d != "object" and c != "DaysLate"])

sel_var = st.sidebar.selectbox("Variable de interés", num_vars,
                               index=num_vars.index("InvoiceAmount")
                               if "InvoiceAmount" in num_vars else 0)

tab_dist, tab_corr = st.tabs(["Distribución", "Correlación"])

with tab_dist:
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.histogram(raw_df, x=sel_var, nbins=40,
                     title=f"Histograma – {sel_var}",
                     color_discrete_sequence=["#3498db"]),
        use_container_width=True,
    )
    col2.plotly_chart(
        px.box(raw_df, y=sel_var, title=f"Boxplot – {sel_var}",
               color_discrete_sequence=["#e74c3c"]),
        use_container_width=True,
    )
    st.dataframe(raw_df[sel_var].describe().to_frame().T.round(2))

with tab_corr:
    corr = raw_df[num_vars + ["DaysLate"]].corr()
    fig  = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                zmin=-1, zmax=1, colorscale="RdBu"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────── 4) FORMULARIO DE PREDICCIÓN ───────────────────
st.header("🧮 Predicción de DaysLate")

with st.form("pred"):
    cL, cR = st.columns(2)
    with cL:
        v_country = st.selectbox("countryCode",
                                 sorted(raw_df["countryCode"].unique()))
        v_amount  = st.number_input("InvoiceAmount", 0.0,
                                    value=float(raw_df["InvoiceAmount"].median()))
        v_disp    = st.selectbox("Disputed", ["No", "Yes"])
        v_paper   = st.selectbox("PaperlessBill", ["Paper", "Electronic"])
        v_dsettle = st.number_input("DaysToSettle", 0,
                                    value=int(raw_df["DaysToSettle"].median()))
    with cR:
        v_idate = st.date_input("InvoiceDate", date(2013, 9, 1))
        v_ddate = st.date_input("DueDate",     date(2013,10, 1))
    submit = st.form_submit_button("Predecir")

if submit:
    sample = {
        "countryCode":     v_country,
        "InvoiceAmount":   v_amount,
        "Disputed":        1 if v_disp=="Yes" else 0,
        "PaperlessBill":   1 if v_paper=="Electronic" else 0,
        "DaysToSettle":    v_dsettle,
        "InvoiceDate_year":  v_idate.year,
        "InvoiceDate_month": v_idate.month,
        "InvoiceDate_day":   v_idate.day,
        "DueDate_year":    v_ddate.year,
        "DueDate_month":   v_ddate.month,
        "DueDate_day":     v_ddate.day,
    }
    X_new = pd.DataFrame([sample])[FEAT_COLS]
    pred  = float(model.predict(X_new)[0])

    if pred < 0:
        st.success(f"✅ Pago estimado **{abs(pred):.1f} días antes** del vencimiento")
    else:
        st.error  (f"🚨 Retraso estimado **{pred:.1f} días**")

st.caption(f"Modelo: MAE {meta['MAE']:.2f} | RMSE {meta['RMSE']:.2f} | R² {meta['R2']:.3f}")

