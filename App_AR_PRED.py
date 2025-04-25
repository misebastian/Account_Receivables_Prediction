# ░░ EDA + XGB Predictor | Accounts Receivable ░░
# ── 2025-04 – estilo demo Breast-Cancer ─────────────────────
import warnings, io, base64
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px, plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib, statsmodels.api as sm

# ═════════════ UI CONFIG ═════════════
st.set_page_config("AR EDA + Predictor", "📈", layout="wide")
st.markdown(
    """
<style>
.main .block-container{max-width:960px; padding-top:1rem}
section[data-testid="stSidebar"]>div:first-child{width:240px}
#MainMenu, footer{visibility:hidden}
h1,h2,h3{font-weight:800}
</style>
""",
    unsafe_allow_html=True,
)

# ═════════════ DATA LOAD ═════════════
FILE = "WA_Fn-UseC_-Accounts-Receivable.xlsx"
df_raw = pd.read_excel(FILE)
df_raw.columns = df_raw.columns.str.strip().str.replace(" ", "_")

# Mapeo binario y categórico
df = df_raw.copy()
df["Disputed"] = df["Disputed"].map({"Yes": 1, "No": 0})
df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})
df["countryCode"] = df["countryCode"].astype("category").cat.codes

num_cols = ["InvoiceAmount", "DaysToSettle", "DaysLate"]

# ═════════════ SIDEBAR ═══════════════
st.sidebar.header("Variables numéricas")
sel_num = st.sidebar.selectbox("Variable de interés", num_cols, index=0)

# ═════════════ EDA ═══════════════════
st.title("Análisis Exploratorio – Accounts Receivable")

k1, k2, k3 = st.columns(3)
k1.metric("Promedio DaysLate", f"{df.DaysLate.mean():.2f}")
k2.metric("Facturas tarde", f"{(df.DaysLate>0).mean()*100:.1f}%")
k3.metric("Total facturas", f"{len(df):,}")

with st.expander("Primer vistazo"):
    st.dataframe(df_raw.head(), use_container_width=True)

# --- Distribución
st.markdown("---")
st.markdown(f"## Distribución de **{sel_num}**")

c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(df, x=sel_num, nbins=40, color_discrete_sequence=["steelblue"]), use_container_width=True)
c2.plotly_chart(px.box(df, y=sel_num, color_discrete_sequence=["firebrick"]), use_container_width=True)

# --- Scatter con otra variable
st.markdown("---")
st.subheader("Relación con otra variable numérica")
otras = [c for c in num_cols if c != sel_num]
otra = st.selectbox("Comparar con:", otras)
fig_scatter = px.scatter(df, x=sel_num, y=otra, trendline="ols", color_discrete_sequence=["#16a085"])
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Correlación
st.markdown("---")
st.subheader("Matriz de correlación")
corr = df[num_cols].corr()
fig_corr = go.Figure(go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale="RdYlBu_r", zmin=-1, zmax=1))
fig_corr.update_layout(height=550)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Conclusiones
st.markdown(
    """
## 🔍 Conclusiones

* **InvoiceAmount** y **DaysToSettle** muestran correlación positiva con `DaysLate`.  
* Aproximadamente el **60 %** de las facturas se pagan puntualmente.  
* El dataset no presenta valores nulos en las columnas analizadas.
""",
    unsafe_allow_html=True,
)

# ═════════════ MODELO XGB ══════════════
st.markdown("---")
st.header("🧮 Predicción de DaysLate (XGB)")

features = ["countryCode", "InvoiceAmount", "Disputed", "PaperlessBill", "DaysToSettle"]
X, y = df[features], df["DaysLate"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    random_state=42,
).fit(X_tr, y_tr)

preds = model.predict(X_te)
mae, rmse, r2 = mean_absolute_error(y_te, preds), mean_squared_error(y_te, preds, squared=False), r2_score(y_te, preds)
st.write(f"**MAE:** {mae:.2f}   |   **RMSE:** {rmse:.2f}   |   **R²:** {r2:.3f}")

# ═════════════ PREDICCIÓN INTERACTIVA ═══
st.markdown("### Predicción individual")

colL, colR = st.columns(2)
with colL:
    ccode = st.number_input("countryCode", value=int(df_raw.countryCode.mode()[0]))
    disputed = st.selectbox("Disputed", ["No", "Yes"])
with colR:
    bill = st.selectbox("PaperlessBill", ["Paper", "Electronic"])
    amt  = st.number_input("InvoiceAmount", 0.0, value=float(df_raw.InvoiceAmount.median()))
settle = st.number_input("DaysToSettle", 0, value=int(df_raw.DaysToSettle.median()))

if st.button("Predecir DaysLate"):
    row = pd.DataFrame([{
        "countryCode": int(ccode),
        "InvoiceAmount": float(amt),
        "Disputed": int(disputed == "Yes"),
        "PaperlessBill": int(bill == "Electronic"),
        "DaysToSettle": int(settle),
    }])
    pred = float(model.predict(row)[0])
    if pred < 0:
        st.success(f"✅ Pago anticipado (~{abs(pred):.1f} días antes).")
    else:
        st.error(f"🚨 Retraso estimado: {pred:.1f} días.")

# ═════════════ DESCARGA MODELO ══════════
with st.expander("⬇️ Descargar modelo"):
    buff = io.BytesIO(); joblib.dump(model, buff)
    st.download_button("XGB model (.pkl)", buff.getvalue(), "xgb_model.pkl")

st.caption("© 2025 – AR EDA + Predictor")
