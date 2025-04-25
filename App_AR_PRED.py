# app.py  –  Streamlit AR Predictor
# ───────────────────────────────────────────────────────────
# Requisitos en requirements.txt
#   streamlit>=1.32
#   pandas numpy plotly openpyxl joblib
#   (xgboost solo lo necesitaste para entrenar)
# ───────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, joblib, json, io, requests
import plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import date

# ------------------------------------------------ CONFIG
st.set_page_config(page_title="📈 AR Predictor",
                   page_icon="📊", layout="wide")
st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child {width: 240px;}
#MainMenu, footer {visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------ 1) DATA & MODEL
DATA_PATH  = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
MODEL_URL  = "https://raw.githubusercontent.com/<TU-USER>/<TU-REPO>/main/ar_pipeline.pkl"
META_URL   = "https://raw.githubusercontent.com/<TU-USER>/<TU-REPO>/main/ar_meta.json"

@st.cache_data(show_spinner=False)
def load_data(fp:Path)->pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ","_")
    df["Disputed"]      = df["Disputed"].map({"Yes":1,"No":0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic":1,"Paper":0})
    for c in ["InvoiceDate","DueDate"]:
        dt = pd.to_datetime(df[c])
        df[[f"{c}_year",f"{c}_month",f"{c}_day"]] = np.c_[dt.dt.year,
                                                          dt.dt.month,
                                                          dt.dt.day]
    return df
raw = load_data(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_model(url:str):
    model  = joblib.load(io.BytesIO(requests.get(url).content))
    meta   = json.loads(requests.get(META_URL).text)
    return model, meta
model, meta = load_model(MODEL_URL)

FEATS = ["countryCode","InvoiceAmount","Disputed","PaperlessBill",
         "DaysToSettle","InvoiceDate_year","InvoiceDate_month",
         "InvoiceDate_day","DueDate_year","DueDate_month","DueDate_day"]

# ------------------------------------------------ 2) KPI HEADER
st.title("📊 Exploratorio – Accounts Receivable")

colA,colB,colC,colD = st.columns(4)
colA.metric("Facturas",&nbsp;f"{len(raw):,}")
colB.metric("Prom. DaysLate", f"{raw['DaysLate'].mean():.2f}")
colC.metric("% a Tiempo",
            f"{(raw['DaysLate']<=0).mean()*100:.1f}%")
colD.metric("% Retraso > 5 días",
            f"{(raw['DaysLate']>5).mean()*100:.1f}%")

st.divider()

# ------------------------------------------------ 3) VARIABLE EXPLORER
num_vars = [c for c,d in zip(raw.columns,raw.dtypes) if d!='object' and c!="DaysLate"]
sel = st.sidebar.selectbox("Variable de interés", sorted(num_vars),
                           index= num_vars.index("InvoiceAmount") )

tab1,tab2 = st.tabs(["Distribución","Correlación"])

with tab1:
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(raw,x=sel,nbins=40,
                                 color_discrete_sequence=["#3498db"],
                                 title=f"Histograma – {sel}"),
                    use_container_width=True)
    c2.plotly_chart(px.box(raw,y=sel,title=f"Boxplot – {sel}",
                           color_discrete_sequence=["#e74c3c"]),
                    use_container_width=True)
    st.dataframe(raw[sel].describe().to_frame().T.round(2))

with tab2:
    corr = raw[num_vars+["DaysLate"]].corr()
    fig  = go.Figure(go.Heatmap(z=corr.values, x=corr.columns,
                                y=corr.columns, colorscale="RdBu",
                                zmin=-1,zmax=1))
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig,use_container_width=True)

st.divider()

# ------------------------------------------------ 4) PREDICTOR
st.header("🧮 Predicción de *DaysLate*")

with st.form("pred"):
    cL,cR = st.columns(2)
    with cL:
        cc   = st.selectbox("countryCode", sorted(raw["countryCode"].unique()))
        inv  = st.number_input("InvoiceAmount",0.0,value=float(raw["InvoiceAmount"].median()))
        disp = st.selectbox("Disputed",["No","Yes"])
        pap  = st.selectbox("PaperlessBill",["Paper","Electronic"])
        dsett= st.number_input("DaysToSettle",0,value=int(raw["DaysToSettle"].median()))
    with cR:
        idate= st.date_input("InvoiceDate",date(2013,9,1))
        ddate= st.date_input("DueDate",    date(2013,10,1))
    go_btn = st.form_submit_button("Predecir")

if go_btn:
    row = {
        "countryCode": cc,
        "InvoiceAmount": inv,
        "Disputed":      1 if disp=="Yes" else 0,
        "PaperlessBill": 1 if pap=="Electronic" else 0,
        "DaysToSettle":  dsett,
        "InvoiceDate_year":  idate.year,
        "InvoiceDate_month": idate.month,
        "InvoiceDate_day":   idate.day,
        "DueDate_year":  ddate.year,
        "DueDate_month": ddate.month,
        "DueDate_day":   ddate.day
    }
    Xnew = pd.DataFrame([row])[FEATS]
    pred = float(model.predict(Xnew)[0])
    if pred<0:
        st.success(f"✅ Pago estimado **{abs(pred):.1f} días antes** del vencimiento.")
    else:
        st.error  (f"🚨 Retraso estimado: **{pred:.1f} días**")

st.caption(f"Modelo MAE {meta['MAE']:.2f}  |  RMSE {meta['RMSE']:.2f}  |  R² {meta['R2']:.3f}")
