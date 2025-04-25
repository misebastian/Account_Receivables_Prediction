# ------------------------------------------------------------
#  Streamlit – Advanced AR Predictor (EDA + XGBRegressor)
#  Author: 2025-04 – adaptado para Streamlit Cloud
# ------------------------------------------------------------
import warnings, io, base64, typing as t
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# visual
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib, statsmodels

# ----------  CONFIGURACIÓN ----------------------------------------------------
st.set_page_config("Account Receivables Prediction", "📈", layout="wide")

st.markdown(
    """
    <style>
        /* enfocar tema oscuro & anchura */
        section[data-testid="stSidebar"] > div:first-child { width:260px; }
        footer, #MainMenu {visibility:hidden;}
        h1, h2, h3 { font-weight: 800;}
    </style>
""",
    unsafe_allow_html=True,
)

# ---------- 1 · CARGA DE DATOS  ----------------------------------------------
FILE = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
if not FILE.exists():
    st.error("❌ Sube el archivo **WA_Fn-UseC_-Accounts-Receivable.xlsx** al directorio raíz del repo.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


raw_df = load_data(FILE)

# ---------- 2 · FEATURE ENGINEERING ------------------------------------------
def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # fechas a datetime
    for c in ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]:
        df[c] = pd.to_datetime(df[c])
    # mapeo binario
    df["Disputed"] = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})

    # componentes de fecha
    for c in ["InvoiceDate", "DueDate", "PaperlessDate", "SettledDate"]:
        df[[f"{c}_year", f"{c}_month", f"{c}_day"]] = df[c].dt.year, df[c].dt.month, df[c].dt.day

    # histórico simple (pagadas / tarde)
    df = df.sort_values(["customerID", "InvoiceDate"]).reset_index(drop=True)

    def _hist(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["is_late"] = (g["DaysLate"] > 0).astype(int)
        g["late_count_agg"] = g["is_late"].shift().fillna(0).cumsum()
        return g

    df = df.groupby("customerID", group_keys=False).apply(_hist)

    drop = ["invoiceNumber", "PaperlessDate", "SettledDate"]  # guardamos customerID para el embedding
    core = df.drop(columns=drop)
    return core, df


engineered_df, df_with_id = engineer(raw_df)

# ---------- 3 · SECCIÓN EDA ---------------------------------------------------
st.title("📊 Exploratorio de Cuentas por Cobrar")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Promedio DaysLate", f"{engineered_df.DaysLate.mean():.2f}")
kpi2.metric("Mediana DaysLate", f"{engineered_df.DaysLate.median():.0f}")
kpi3.metric("Pagos a Tiempo", f"{(engineered_df.DaysLate<=0).mean()*100:.1f}%")

with st.expander("Descriptivo completo"):
    st.dataframe(engineered_df.describe().T.round(2), use_container_width=True)

num_cols = engineered_df.select_dtypes(include=np.number).columns.tolist()
default_idx = num_cols.index("InvoiceAmount") if "InvoiceAmount" in num_cols else 0
sel = st.sidebar.selectbox("Variable numérica a explorar", num_cols, index=default_idx)

# --- histogram + boxplot
col1, col2 = st.columns(2)
col1.plotly_chart(px.histogram(engineered_df, x=sel, nbins=40, color_discrete_sequence=["#2a9df4"]), use_container_width=True)
col2.plotly_chart(px.box(engineered_df, y=sel, color_discrete_sequence=["#e74c3c"]), use_container_width=True)

# --- scatter (sólo si sel ≠ DaysLate y es numérica)
st.subheader(f"Relación {sel} vs DaysLate")
if sel != "DaysLate":
    trend = "ols" if np.issubdtype(engineered_df[sel].dtype, np.number) else None
    scat_df = engineered_df[[sel, "DaysLate"]].dropna()
    fig_scat = px.scatter(
        scat_df.sample(min(3000, len(scat_df))),
        x=sel,
        y="DaysLate",
        opacity=0.6,
        trendline=trend,
        color_discrete_sequence=["#16a085"],
    )
    st.plotly_chart(fig_scat, use_container_width=True)
else:
    st.info("Selecciona otra variable distinta de **DaysLate** para ver la dispersión.")

# --- matriz de correlación
with st.expander("Matriz de correlación"):
    corr = engineered_df[num_cols].corr()
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdYlBu_r",
            zmin=-1,
            zmax=1,
        )
    )
    fig_corr.update_layout(height=650)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ---------- 4 · EMBEDDING + XGB (con caché) ----------------------------------
@st.cache_data(show_spinner=True)
def train_model(df_core: pd.DataFrame, df_all: pd.DataFrame):
    # ---- embedding
    cust_idx = df_all.customerID.astype("category").cat.codes
    n_cust = cust_idx.nunique()
    emb_dim = int(np.ceil(np.log2(n_cust)))
    tf.random.set_seed(42)
    emb_model = models.Sequential(
        [
            layers.Embedding(n_cust, emb_dim, input_length=1),
            layers.Flatten(),
            layers.Dense(8, activation="relu"),
            layers.Dense(1),
        ]
    )
    emb_model.compile(optimizer="adam", loss="mae")
    emb_model.fit(cust_idx, df_all.DaysLate, epochs=15, batch_size=256, verbose=0)
    emb_weights = emb_model.layers[0].get_weights()[0]
    emb_cols = [f"cust_emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_weights, columns=emb_cols).assign(cust_idx=np.arange(n_cust))
    df_all = df_all.assign(cust_idx=cust_idx)
    full = df_core.reset_index(drop=True).join(emb_df.set_index("cust_idx"), on="cust_idx")

    # ---- split temporal
    full["InvoiceDate"] = pd.to_datetime(raw_df.InvoiceDate)
    cutoff = pd.Timestamp("2013-07-01")
    train_mask = full.InvoiceDate < cutoff

    X_train = full.loc[train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    X_test = full.loc[~train_mask].drop(columns=["DaysLate", "InvoiceDate"])
    y_train = full.loc[train_mask, "DaysLate"]
    y_test = full.loc[~train_mask, "DaysLate"]

    scale_cols = [c for c in X_train.columns if not c.startswith("cust_emb_")]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    # ---- XGB
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    metrics = dict(
        MAE=mean_absolute_error(y_test, preds),
        RMSE=mean_squared_error(y_test, preds, squared=False),
        R2=r2_score(y_test, preds),
    )
    return xgb, scaler, emb_cols, metrics


with st.spinner("Entrenando modelo…"):
    best_model, scaler, emb_cols, m = train_model(engineered_df, df_with_id)

st.success(f"Modelo entrenado – MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | R² {m['R2']:.3f}")

# ---------- 5 · PREDICCIÓN INTERACTIVA ---------------------------------------
st.markdown("---")
st.header("🔮 Predicción interactiva de *DaysLate*")

country_codes = sorted(raw_df.countryCode.unique())
customers = sorted(raw_df.customerID.unique())

with st.form("pred_form"):
    left, right = st.columns(2)
    with left:
        cc = st.selectbox("countryCode", country_codes)
        disputed = st.selectbox("Disputed", ["No", "Yes"])
        paperless = st.selectbox("PaperlessBill", ["Paper", "Electronic"])
        cust = st.selectbox("customerID", customers)
    with right:
        inv_amt = st.number_input("InvoiceAmount", 0.0, value=float(raw_df.InvoiceAmount.median()))
        days_settle = st.number_input("DaysToSettle", 0, value=int(raw_df.DaysToSettle.median()))
        inv_date = st.date_input("InvoiceDate", value=date(2013, 9, 1))
        due_date = st.date_input("DueDate", value=date(2013, 10, 1))
    ok = st.form_submit_button("Predecir")

if ok:
    # fila nueva
    row: dict[str, t.Any] = dict(
        countryCode=cc,
        InvoiceAmount=inv_amt,
        Disputed=1 if disputed == "Yes" else 0,
        PaperlessBill=1 if paperless == "Electronic" else 0,
        DaysToSettle=days_settle,
        cust_idx=df_with_id.loc[df_with_id.customerID == cust, "cust_idx"].iloc[0],
    )
    # fechas -> componentes
    for label, dval in [("InvoiceDate", inv_date), ("DueDate", due_date)]:
        d = pd.to_datetime(dval)
        row[f"{label}_year"], row[f"{label}_month"], row[f"{label}_day"] = d.year, d.month, d.day
    # agregar embeddings
    emb_vals = engineered_df.join(df_with_id[["cust_idx"]]).merge(
        pd.concat([df_with_id["cust_idx"], engineered_df[emb_cols]], axis=1).drop_duplicates(),
        on="cust_idx",
        how="left",
    )
    row.update(emb_vals.loc[emb_vals.cust_idx == row["cust_idx"], emb_cols].iloc[0].to_dict())

    X_new = pd.DataFrame([row])[best_model.feature_names_in_]
    X_new[[c for c in X_new.columns if c in scaler.feature_names_in_]] = scaler.transform(
        X_new[[c for c in X_new.columns if c in scaler.feature_names_in_]]
    )
    pred = float(best_model.predict(X_new)[0])

    if pred < 0:
        st.success(f"✅ Pagará **{abs(pred):.1f} días antes** del vencimiento.")
    else:
        st.error(f"🚨 Se retrasará **{pred:.1f} días**.")

# ---------- 6 · DESCARGA  -----------------------------------------------------
with st.expander("⬇️ Descargar modelo y scaler"):
    buf_m, buf_s = io.BytesIO(), io.BytesIO()
    joblib.dump(best_model, buf_m)
    joblib.dump(scaler, buf_s)
    b64m = base64.b64encode(buf_m.getvalue()).decode()
    b64s = base64.b64encode(buf_s.getvalue()).decode()
    st.markdown(
        f'<a download="xgb_model.pkl" href="data:application/octet-stream;base64,{b64m}">Modelo XGB</a> | '
        f'<a download="scaler.pkl" href="data:application/octet-stream;base64,{b64s}">Scaler</a>',
        unsafe_allow_html=True,
    )

st.caption("© 2025 – AR Predictor Demo")
