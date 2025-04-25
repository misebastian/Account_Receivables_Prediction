# ░░ Account-Receivables Predictor – versión compacta ░░
# ----------------------------------------------------
import warnings, io, base64, typing as t
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

import plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib, statsmodels.api as sm    # ← línea OLS del hist/box

# ═════════════ CONFIG ═══════════════════════════════
st.set_page_config("AR Predictor", "📈", layout="wide")
st.markdown(
    """
    <style>
        .main .block-container{max-width: 980px; padding-top:1rem}
        section[data-testid="stSidebar"] > div:first-child{width:240px}
        #MainMenu,footer{visibility:hidden}
        h1,h2,h3{font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════ 1 · DATA LOAD ════════════════════════
FILE = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
if not FILE.exists():
    st.error("❌ Falta **WA_Fn-UseC_-Accounts-Receivable.xlsx**.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


raw = load_data(FILE)

# ═════════════ 2 · FEATURE ENGINEERING ══════════════
def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    for c in ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]:
        df[c] = pd.to_datetime(df[c])

    df["Disputed"] = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})

    for c in ["InvoiceDate", "DueDate"]:
        df[f"{c}_year"] = df[c].dt.year
        df[f"{c}_month"] = df[c].dt.month
        df[f"{c}_day"] = df[c].dt.day

    df = df.sort_values(["customerID", "InvoiceDate"]).reset_index(drop=True)

    def _hist(g):
        g = g.copy()
        g["is_late"] = (g["DaysLate"] > 0).astype(int)
        g["late_count_acc"] = g["is_late"].shift().fillna(0).cumsum()
        return g

    df = df.groupby("customerID", group_keys=False).apply(_hist)

    core = df.drop(columns=["invoiceNumber", "PaperlessDate", "SettledDate"])
    return core, df


core_df, df_id = engineer(raw)

# ═════════════ 3 · EDA ══════════════════════════════
st.title("📊 EDA – Cuentas por Cobrar")

k1, k2, k3 = st.columns(3)
k1.metric("Promedio DaysLate", f"{core_df.DaysLate.mean():.2f}")
k2.metric("Mediana", f"{core_df.DaysLate.median():.0f}")
k3.metric("Pagos a tiempo", f"{(core_df.DaysLate<=0).mean()*100:.1f}%")

with st.expander("Primer vistazo"):
    st.dataframe(core_df.head(), use_container_width=True)

num_cols = core_df.select_dtypes(np.number).columns.tolist()
sel = st.sidebar.selectbox("Variable numérica", num_cols, index=num_cols.index("InvoiceAmount"))

# Histograma y boxplot
c1, c2 = st.columns(2)
c1.plotly_chart(px.histogram(core_df, x=sel, nbins=40, color_discrete_sequence=["#3498db"]), use_container_width=True)
c2.plotly_chart(px.box(core_df, y=sel, color_discrete_sequence=["#e74c3c"]), use_container_width=True)

with st.expander("Matriz de correlación"):
    corr = core_df[num_cols].corr()
    fig = go.Figure(go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
### 🔍 Conclusiones

1. `InvoiceAmount` y `DaysToSettle` son los mejores predictores de **DaysLate**.  
2. ~60 % de las facturas se pagan en o antes de la fecha de vencimiento.  
3. El dataset no contiene nulos ⇒ no se requiere imputación.
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═════════════ 4 · EMBEDDING + XGB ═════════════════
@st.cache_data(show_spinner="Entrenando modelo…")
def build_model(core: pd.DataFrame, full: pd.DataFrame):
    cust_idx = full.customerID.astype("category").cat.codes
    n_cust, dim = cust_idx.nunique(), int(np.ceil(np.log2(cust_idx.nunique())))

    tf.random.set_seed(42)
    emb_net = models.Sequential(
        [layers.Embedding(n_cust, dim, input_length=1),
         layers.Flatten(),
         layers.Dense(8, activation="relu"),
         layers.Dense(1)]
    )
    emb_net.compile("adam", "mae")
    emb_net.fit(cust_idx, full.DaysLate, epochs=15, batch_size=256, verbose=0)

    emb = emb_net.layers[0].get_weights()[0]
    emb_cols = [f"cust_emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(emb, columns=emb_cols).assign(cust_idx=np.arange(n_cust))

    full = full.assign(cust_idx=cust_idx).merge(emb_df, on="cust_idx")
    data = core.reset_index(drop=True).join(full[emb_cols])

    cutoff = pd.Timestamp("2013-07-01")
    data["InvoiceDate"] = pd.to_datetime(raw.InvoiceDate)
    train_mask = data.InvoiceDate < cutoff

    X_train = data.loc[train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    X_test  = data.loc[~train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    y_train, y_test = data.loc[train_mask, "DaysLate"], data.loc[~train_mask, "DaysLate"]

    scale_cols = [c for c in X_train.columns if not c.startswith("cust_emb_")]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols].astype("float64"))
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols].astype("float64"))

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
    metrics = dict(MAE=mean_absolute_error(y_test, preds),
                   RMSE=mean_squared_error(y_test, preds, squared=False),
                   R2=r2_score(y_test, preds))
    return xgb, scaler, emb_cols, metrics


model, scaler, emb_cols, met = build_model(core_df, df_id)
st.success(f"Modelo: MAE {met['MAE']:.2f} | RMSE {met['RMSE']:.2f} | R² {met['R2']:.3f}")

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
        inv_d = st.date_input("InvoiceDate", date(2013, 9, 1))
        due_d = st.date_input("DueDate", date(2013, 10, 1))
    ok = st.form_submit_button("Predecir")

if ok:
    row: dict[str, t.Any] = dict(
        countryCode=cc,
        InvoiceAmount=amt,
        Disputed=int(disputed == "Yes"),
        PaperlessBill=int(paperless == "Electronic"),
        DaysToSettle=settle,
        cust_idx=df_id.loc[df_id.customerID == cust, "cust_idx"].iloc[0],
    )
    for label, d in [("InvoiceDate", inv_d), ("DueDate", due_d)]:
        d = pd.to_datetime(d)
        row[f"{label}_year"], row[f"{label}_month"], row[f"{label}_day"] = d.year, d.month, d.day

    row.update(df_id.drop_duplicates("cust_idx").set_index("cust_idx")[emb_cols].loc[row["cust_idx"]].to_dict())

    X_new = pd.DataFrame([row])[model.feature_names_in_]
    X_new[scaler.feature_names_in_] = scaler.transform(X_new[scaler.feature_names_in_].astype("float64"))
    pred = float(model.predict(X_new)[0])

    st.success(f"Resultado: {pred:+.1f} días {'antes' if pred<0 else 'después'} del vencimiento")

# ═════════════ 6 · DESCARGAS ═════════════=
with st.expander("⬇️ Descargar artefactos"):
    bm, bs = io.BytesIO(), io.BytesIO()
    joblib.dump(model, bm); joblib.dump(scaler, bs)
    st.download_button("Modelo XGB", bm.getvalue(), "xgb_model.pkl")
    st.download_button("Scaler", bs.getvalue(), "scaler.pkl")

st.caption("© 2025 | AR Predictor compact")
