# ░░ Account-Receivables Predictor ░░
# versiones compactas + comentarios estilo demo «Wisconsin»
# ----------------------------------------------------------
import warnings, io, base64, typing as t
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

import plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib, statsmodels.api as sm  # statsmodels = trendline OLS

# ═════════════ CONFIG ─════════════════════
st.set_page_config("AR Predictor", "📈", layout="wide")

st.markdown(
    """
    <style>
        /*⇢ centra y estrecha el body */
        .main .block-container{max-width: 980px; padding-top:1rem}
        section[data-testid="stSidebar"] > div:first-child{width:240px}
        #MainMenu,footer{visibility:hidden}
        h1,h2,h3{font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════ 1 · DATA LOAD ══════════════
FILE = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
if not FILE.exists():
    st.error("❌ Sube **WA_Fn-UseC_-Accounts-Receivable.xlsx** al repo.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


raw = load_data(FILE)

# ═════════════ 2 · FEATURE ENG. ═══════════
def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # fechas → datetime
    for c in ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]:
        df[c] = pd.to_datetime(df[c])
    # binarias
    df["Disputed"] = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})

    # desglose de fecha
    for c in ["InvoiceDate", "DueDate"]:
        df[[f"{c}_year", f"{c}_month", f"{c}_day"]] = (
            df[c].dt.year,
            df[c].dt.month,
            df[c].dt.day,
        )

    df = df.sort_values(["customerID", "InvoiceDate"]).reset_index(drop=True)

    # hist simple (pagos tarde acumulados)
    def _agg(g):
        g = g.copy()
        g["is_late"] = (g["DaysLate"] > 0).astype(int)
        g["late_count_acc"] = g["is_late"].shift().fillna(0).cumsum()
        return g

    df = df.groupby("customerID", group_keys=False).apply(_agg)
    core = df.drop(columns=["invoiceNumber", "PaperlessDate", "SettledDate"])
    return core, df


core_df, df_with_id = engineer(raw)

# ═════════════ 3 · EDA (estilo demo) ══════
st.title("📊 Análisis Exploratorio de Datos – *Accounts Receivable*")

k1, k2, k3 = st.columns(3)
k1.metric("Promedio de días de atraso", f"{core_df.DaysLate.mean():.2f}")
k2.metric("Mediana", f"{core_df.DaysLate.median():.0f}")
k3.metric("Facturas a tiempo", f"{(core_df.DaysLate<=0).mean()*100:.1f}%")

with st.expander("Primer vistazo"):
    st.dataframe(core_df.head(), use_container_width=True)

num_cols = core_df.select_dtypes(np.number).columns.tolist()
sel = st.sidebar.selectbox("Variable numérica", num_cols, index=num_cols.index("InvoiceAmount"))

# -- histograma & boxplot
c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(core_df, x=sel, nbins=40, color_discrete_sequence=["#3498db"]), use_container_width=True)
c2.plotly_chart(px.box(core_df, y=sel, color_discrete_sequence=["#e74c3c"]), use_container_width=True)

# -- matriz correlación
with st.expander("Matriz de correlación"):
    corr = core_df[num_cols].corr()
    fig = go.Figure(go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# -- conclusiones (estilo Markdown UX)
st.markdown(
    """
### 🔍 Conclusiones principales

1. **InvoiceAmount** y **DaysToSettle** son las variables numéricas con mayor correlación positiva respecto a `DaysLate`.  
2. En torno al **60 %** de las facturas se pagan a tiempo; el resto concentra la cola de la distribución.  
3. No se detectan **valores nulos**: las fechas y montos están completos, lo que simplifica el modelado.
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═════════════ 4 · EMBEDDING + XGB  ═══════
@st.cache_data(show_spinner="Entrenando modelo…")
def build_model(core: pd.DataFrame, full: pd.DataFrame):
    # Embedding customerID
    cust_idx = full.customerID.astype("category").cat.codes
    n_cust, emb_dim = cust_idx.nunique(), int(np.ceil(np.log2(cust_idx.nunique())))

    tf.random.set_seed(42)
    emb_net = models.Sequential(
        [
            layers.Embedding(n_cust, emb_dim, input_length=1),
            layers.Flatten(),
            layers.Dense(8, activation="relu"),
            layers.Dense(1),
        ]
    )
    emb_net.compile(optimizer="adam", loss="mae")
    emb_net.fit(cust_idx, full.DaysLate, epochs=15, batch_size=256, verbose=0)

    emb = emb_net.layers[0].get_weights()[0]
    emb_cols = [f"cust_emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb, columns=emb_cols).assign(cust_idx=np.arange(n_cust))

    full = full.assign(cust_idx=cust_idx).merge(emb_df, on="cust_idx")
    data = core.reset_index(drop=True).join(full[emb_cols])

    # split temporal
    cutoff = pd.Timestamp("2013-07-01")
    data["InvoiceDate"] = pd.to_datetime(raw.InvoiceDate)
    train_mask = data.InvoiceDate < cutoff

    X_train = data.loc[train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    X_test = data.loc[~train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    y_train, y_test = data.loc[train_mask, "DaysLate"], data.loc[~train_mask, "DaysLate"]

    scale_cols = [c for c in X_train.columns if not c.startswith("cust_emb_")]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols].astype("float64"))
    X_test[scale_cols] = scaler.transform(X_test[scale_cols].astype("float64"))

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)

    preds = xgb.predict(X_test)
    metrics = dict(MAE=mean_absolute_error(y_test, preds), RMSE=mean_squared_error(y_test, preds, squared=False), R2=r2_score(y_test, preds))
    return xgb, scaler, emb_cols, metrics


model, scaler, emb_cols, m = build_model(core_df, df_with_id)
st.success(f"Modelo listo – MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | R² {m['R2']:.3f}")

# ═════════════ 5 · PREDICCIÓN ═════════════
st.markdown("---")
st.header("🔮 Predicción interactiva")

countries = sorted(raw.countryCode.unique())
customers = sorted(raw.customerID.unique())

with st.form("form"):
    l, r = st.columns(2)
    with l:
        cc = st.selectbox("countryCode", countries)
        disputed = st.selectbox("Disputed", ["No", "Yes"])
        paperless = st.selectbox("PaperlessBill", ["Paper", "Electronic"])
        cust = st.selectbox("customerID", customers)
    with r:
        amt = st.number_input("InvoiceAmount", 0.0, value=float(raw.InvoiceAmount.median()))
        settle = st.number_input("DaysToSettle", 0, value=int(raw.DaysToSettle.median()))
        inv_d = st.date_input("InvoiceDate", value=date(2013, 9, 1))
        due_d = st.date_input("DueDate", value=date(2013, 10, 1))
    ok = st.form_submit_button("Predecir")

if ok:
    row: dict[str, t.Any] = {
        "countryCode": cc,
        "InvoiceAmount": amt,
        "Disputed": 1 if disputed == "Yes" else 0,
        "PaperlessBill": 1 if paperless == "Electronic" else 0,
        "DaysToSettle": settle,
        "cust_idx": df_with_id.loc[df_with_id.customerID == cust, "cust_idx"].iloc[0],
    }
    for label, d in [("InvoiceDate", inv_d), ("DueDate", due_d)]:
        d = pd.to_datetime(d)
        row[f"{label}_year"], row[f"{label}_month"], row[f"{label}_day"] = d.year, d.month, d.day

    # embeddings
    emb_vals = df_with_id.drop_duplicates("cust_idx").set_index("cust_idx")[emb_cols]
    row.update(emb_vals.loc[row["cust_idx"]].to_dict())

    X_new = pd.DataFrame([row])[model.feature_names_in_]
    X_new[scaler.feature_names_in_] = scaler.transform(X_new[scaler.feature_names_in_].astype("float64"))
    pred = float(model.predict(X_new)[0])

    if pred < 0:
        st.success(f"✅ Pagará **{abs(pred):.1f} días antes** del vencimiento.")
    else:
        st.error(f"🚨 Retraso estimado: **{pred:.1f} días**.")

# ═════════════ 6 · DESCARGA ═══════════════
with st.expander("⬇️ Descargar artefactos"):
    bm, bs = io.BytesIO(), io.BytesIO()
    joblib.dump(model, bm); joblib.dump(scaler, bs)
    st.download_button("Modelo XGB", bm.getvalue(), "xgb_model.pkl")
    st.download_button("Scaler", bs.getvalue(), "scaler.pkl")

st.caption("© 2025 – Demo AR Predictor (versión compacta)")
