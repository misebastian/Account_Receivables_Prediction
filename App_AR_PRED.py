# ---------------------------------------------------------------
# Streamlit app – EDA + XGBRegressor (DaysLate) – v3 aesthetic
# ---------------------------------------------------------------
#  🖥  Twenty‑Apr‑2025 – cambia: EDA primero (más visual) y predicción al final.
# ---------------------------------------------------------------
# pip install pandas numpy scikit-learn xgboost tensorflow-cpu plotly streamlit
# ---------------------------------------------------------------

import warnings, io, base64
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# -----------------------------------------------------------------------------
# CONFIG & STYLE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AR Predictor", layout="wide", page_icon="📈")

st.markdown("""
<style>
/* estrechar el sidebar */
section[data-testid="stSidebar"] > div:first-child {width: 260px;}
/* esconder watermark */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1 · LOAD RAW DATA
# -----------------------------------------------------------------------------
FILE = Path("WA_Fn-UseC_-Accounts-Receivable.xlsx")
if not FILE.exists():
    st.error("❌ Fichero 'WA_Fn-UseC_-Accounts-Receivable.xlsx' no encontrado.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_raw(fp):
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df
raw_df = load_raw(FILE)

# -----------------------------------------------------------------------------
# 2 · FEATURE ENGINEERING (4‑month window)
# -----------------------------------------------------------------------------

def engineer(df: pd.DataFrame, window=4):
    df = df.copy()
    for c in ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]:
        df[c] = pd.to_datetime(df[c])
    df["Disputed"] = df["Disputed"].map({"Yes": 1, "No": 0})
    df["PaperlessBill"] = df["PaperlessBill"].map({"Electronic": 1, "Paper": 0})

    for c in ["InvoiceDate", "DueDate", "PaperlessDate", "SettledDate"]:
        df[f"{c}_year"] = df[c].dt.year
        df[f"{c}_month"] = df[c].dt.month
        df[f"{c}_day"] = df[c].dt.day

    df = df.sort_values(["customerID", "InvoiceDate"]).reset_index(drop=True)
    def _agg(g):
        g = g.copy()
        g["paid"] = (g["DaysLate"] <= 0).astype(int)
        g["late"] = (g["DaysLate"] > 0).astype(int)
        g["tot_paid"] = g["paid"].shift().fillna(0).cumsum()
        g["tot_late"] = g["late"].shift().fillna(0).cumsum()
        return g
    df = df.groupby("customerID", group_keys=False).apply(_agg)

    drop_cols = ["invoiceNumber", "customerID", "PaperlessDate", "SettledDate"]
    return df[[c for c in df.columns if c not in drop_cols]], df

engineered_df, df_w_id = engineer(raw_df)

# -----------------------------------------------------------------------------
# 3 · EDA VISUALS  (FIRST SECTION)
# -----------------------------------------------------------------------------

st.title("📊 Exploratorio de Cuentas por Cobrar")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Promedio DaysLate", f"{engineered_df['DaysLate'].mean():.2f}")
kpi2.metric("Mediana DaysLate", f"{engineered_df['DaysLate'].median():.0f}")
kpi3.metric("% Facturas a Tiempo", f"{(engineered_df['DaysLate']<=0).mean()*100:.1f}%")

with st.expander("Descriptivo completo", expanded=False):
    st.dataframe(engineered_df.describe().T.round(2))

num_cols = engineered_df.select_dtypes(include=[np.number]).columns.tolist()
sel = st.sidebar.selectbox("Variable numérica", num_cols, index=num_cols.index("InvoiceAmount"))

hist = px.histogram(engineered_df, x=sel, nbins=40, color_discrete_sequence=["#2a9df4"])
box = px.box(engineered_df, y=sel, color_discrete_sequence=["#e74c3c"])
col1, col2 = st.columns(2)
col1.plotly_chart(hist, use_container_width=True)
col2.plotly_chart(box, use_container_width=True)

st.subheader(f"Relación {sel} vs DaysLate")
scat = px.scatter(engineered_df.sample(min(3000,len(engineered_df))), x=sel, y="DaysLate", opacity=0.6, trendline="ols", color_discrete_sequence=["#16a085"])
st.plotly_chart(scat, use_container_width=True)

with st.expander("Matriz de correlación"):
    corr = engineered_df[num_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdYlBu_r", zmin=-1, zmax=1))
    fig_corr.update_layout(height=650)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4 · EMBEDDING customerID + MODEL TRAINING (happens backstage)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_dataset(df_core, df_all, dim=None):
    cust_idx = df_all["customerID"].astype("category").cat.codes
    n_cust = cust_idx.nunique()
    if dim is None:
        dim = int(np.ceil(np.log2(n_cust)))
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=(1,), dtype="int32"),
        layers.Embedding(n_cust, dim),
        layers.Flatten(),
        layers.Dense(8, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(cust_idx, df_all["DaysLate"], epochs=20, batch_size=256, validation_split=0.1, verbose=0)
    emb = model.layers[1].get_weights()[0]
    emb_cols = [f"cust_emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df["cust_idx"] = np.arange(n_cust)
    df_all["cust_idx"] = cust_idx
    merged = df_all.merge(emb_df, on="cust_idx", how="left")
    final = pd.concat([df_core.reset_index(drop=True), merged[emb_cols]], axis=1)
    return final, emb_cols

with st.spinner("Entrenando embedding y modelo XGB…"):
    full_df, emb_cols = build_dataset(engineered_df, df_w_id)

    # temporal split
    full_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"])
    cutoff = pd.Timestamp("2013-07-01")
    mask = full_df["InvoiceDate"] < cutoff
    X_train = full_df.loc[mask].drop(columns=["DaysLate", "InvoiceDate"])
    X_test  = full_df.loc[~mask].drop(columns=["DaysLate", "InvoiceDate"])
    y_train = full_df.loc[mask, "DaysLate"]
    y_test  = full_df.loc[~mask, "DaysLate"]

    scale_cols = [c for c in X_train.columns if not c.startswith("cust_emb_")]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    param_grid = {"n_estimators":[600], "max_depth":[6], "learning_rate":[0.05], "subsample":[0.8]}
    model = GridSearchCV(XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1), param_grid, cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error", n_jobs=-1)
    model.fit(X_train, y_train)
    best = model.best_estimator_
    y_pred = best.predict(X_test)
    MAE, RMSE, R2 = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False), r2_score(y_test, y_pred)

st.success(f"Modelo listo – MAE: {MAE:.2f}  |  RMSE: {RMSE:.2f}  |  R²: {R2:.3f}")

# -----------------------------------------------------------------------------
# 5 · PREDICTION SECTION (FINAL)
# -----------------------------------------------------------------------------

st.markdown("---")
st.header("🔮 Predicción interactiva de DaysLate")

country_codes = sorted(raw_df["countryCode"].unique())

with st.form("pred_form"):
    colL, colR = st.columns(2)
    with colL:
        cc = st.selectbox("countryCode", country_codes)
        disputed = st.selectbox("Disputed", ["No", "Yes"])
        paperless = st.selectbox("PaperlessBill", ["Paper", "Electronic"])
        cust_id = st.selectbox("customerID", sorted(raw_df["customerID"].unique()))
    with colR:
        inv_amt = st.number_input("InvoiceAmount", min_value=0.0, value=float(raw_df["InvoiceAmount"].median()), step=1.0)
        days_settle = st.number_input("DaysToSettle", min_value=0, value=int(raw_df["DaysToSettle"].median()), step=1)
        inv_date = st.date_input("InvoiceDate", value=date(2013,9,1))
        due_date = st.date_input("DueDate", value=date(2013,10,1))
    submitted = st.form_submit_button("Predecir")

if submitted:
    row = {
        "countryCode": cc,
        "InvoiceAmount": inv_amt,
        "Disputed": 1 if disputed=="Yes" else 0,
        "PaperlessBill": 1 if paperless=="Electronic" else 0,
        "DaysToSettle": days_settle,
        "InvoiceDate": pd.to_datetime(inv_date),
        "DueDate": pd.to_datetime(due_date),
        "cust_idx": df_w_id.loc[df_w_id["customerID"]==cust_id, "cust_idx"].iloc[0]
    }
    for c in ["InvoiceDate", "DueDate"]:
        row[f"{c}_year"] = pd.to_datetime(row[c]).year
        row[f"{c}_month"] = pd.to_datetime(row[c]).month
        row[f"{c}_day"] = pd.to_datetime(row[c]).day
    for col in emb_cols:
        row[col] = full_df.loc[df_w_id["customerID"]==cust_id, col].iloc[0]
    X_new = pd.DataFrame([row])[X_train.columns]
    X_new[scale_cols] = scaler.transform(X_new[scale_cols])
    pred = best.predict(X_new)[0]
    if pred < 0:
        st.success(f"✅ Probable pago {abs(pred):.1f} días **antes** del vencimiento.")
    else:
        st.error(f"🚨 Probable retraso de {pred:.1f} días.")

# -----------------------------------------------------------------------------
# 6 · DOWNLOAD BUTTONS
# -----------------------------------------------------------------------------
with st.expander("Descargar modelo y scaler"):
    buf_m, buf_s = io.BytesIO(), io.BytesIO()
    joblib.dump(best, buf_m)
    joblib.dump(scaler, buf_s)
    href_m = f'<a download="xgb_model.pkl" href="data:application/octet-stream;base64,{base64.b64encode(buf_m.getvalue()).decode()}">Modelo XGB</a>'
    href_s = f'<a download="scaler.pkl" href="data:application/octet-stream;base64,{base64.b64encode(buf_s.getvalue()).decode()}">Scaler</a>'
    st.markdown(href_m, unsafe_allow_html=True); st.markdown(" | ", unsafe_allow_html=True); st.markdown(href_s, unsafe_allow_html=True)

st.caption("© 2025 – Demo AR Predictor | v3")

