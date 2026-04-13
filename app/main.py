"""
app/main.py
============================================================
PHASE: DEPLOYMENT — STREAMLIT WEATHER PREDICTION APP
Delhi Weather Forecasting – Ensemble ML Model
============================================================
Run:  streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Weather Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #5B7FA6;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F, #2E6DA4);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #FFD700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .info-box {
        background: #EBF4FF;
        border-left: 4px solid #2E6DA4;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F, #2E6DA4);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        xgb_model   = joblib.load(os.path.join(base, "models/xgboost_model.pkl"))
        lgb_model   = joblib.load(os.path.join(base, "models/lightgbm_model.pkl"))
        weights     = joblib.load(os.path.join(base, "models/ensemble_weights.pkl"))
        meta        = joblib.load(os.path.join(base, "models/feature_meta.pkl"))
        return xgb_model, lgb_model, weights, meta
    except FileNotFoundError:
        return None, None, None, None


xgb_model, lgb_model, weights, meta = load_models()
models_loaded = xgb_model is not None


# ── Prediction Helper ─────────────────────────────────────────
def build_input_row(temp, humidity, wind_speed, pressure,
                    temp_yesterday, temp_2days_ago, features):
    """
    Build a single-row DataFrame with all engineered features.
    Uses simplified estimates for rolling/lag features when only
    current day values are provided.
    """
    import datetime
    today = datetime.date.today()
    month = today.month
    doy   = today.timetuple().tm_yday

    row = {}
    for f in features:
        row[f] = 0.0  # default

    # Base meteorological inputs
    row['meantemp']     = temp
    row['humidity']     = humidity
    row['wind_speed']   = wind_speed
    row['meanpressure'] = pressure

    # Time features
    row['month']        = month
    row['day']          = today.day
    row['day_of_year']  = doy
    row['day_of_week']  = today.weekday()
    row['season']       = {12:1,1:1,2:1,3:2,4:2,5:2,
                           6:3,7:3,8:3,9:4,10:4,11:4}[month]
    row['month_sin']    = np.sin(2 * np.pi * month / 12)
    row['month_cos']    = np.cos(2 * np.pi * month / 12)
    row['doy_sin']      = np.sin(2 * np.pi * doy / 365)
    row['doy_cos']      = np.cos(2 * np.pi * doy / 365)

    # Lag features
    row['temp_lag1']    = temp_yesterday
    row['temp_lag2']    = temp_2days_ago
    row['temp_lag3']    = temp_2days_ago      # approximation
    row['temp_lag7']    = (temp + temp_yesterday + temp_2days_ago) / 3
    row['humidity_lag1'] = humidity
    row['humidity_lag2'] = humidity
    row['humidity_lag3'] = humidity
    row['humidity_lag7'] = humidity
    row['pressure_lag1'] = pressure
    row['pressure_lag2'] = pressure
    row['pressure_lag3'] = pressure
    row['pressure_lag7'] = pressure
    row['wind_lag1']    = wind_speed
    row['wind_lag2']    = wind_speed
    row['wind_lag3']    = wind_speed
    row['wind_lag7']    = wind_speed

    # Rolling features
    avg3  = (temp + temp_yesterday + temp_2days_ago) / 3
    avg7  = avg3
    avg14 = avg3
    for window, avg in [(3, avg3), (7, avg7), (14, avg14)]:
        row[f'temp_roll_mean{window}'] = avg
        row[f'temp_roll_std{window}']  = abs(temp - temp_yesterday) * 0.5
        row[f'hum_roll_mean{window}']  = humidity

    row['temp_ewm7']  = 0.7 * temp + 0.3 * temp_yesterday
    row['temp_ewm14'] = 0.6 * temp + 0.4 * temp_yesterday

    # Interaction features
    row['heat_index']     = temp * humidity / 100
    row['pressure_delta'] = pressure - row['pressure_lag1']
    row['temp_delta']     = temp - temp_yesterday
    row['wind_chill']     = temp - 0.5 * wind_speed

    # Build DataFrame with correct column order
    df = pd.DataFrame([row])
    df = df.reindex(columns=features, fill_value=0.0)
    return df


def predict_temperature(temp, humidity, wind_speed, pressure,
                         temp_yesterday, temp_2days_ago):
    if not models_loaded:
        return None, None, None

    features = meta['features']
    X = build_input_row(temp, humidity, wind_speed, pressure,
                        temp_yesterday, temp_2days_ago, features)

    xgb_pred = xgb_model.predict(X)[0]
    lgb_pred = lgb_model.predict(X)[0]
    w_xgb    = weights['w_xgb']
    w_lgb    = weights['w_lgb']
    ens_pred = w_xgb * xgb_pred + w_lgb * lgb_pred

    return xgb_pred, lgb_pred, ens_pred


def temp_to_emoji(t):
    if t < 10:   return "🥶 Very Cold"
    if t < 18:   return "🌥️ Cool"
    if t < 26:   return "🌤️ Pleasant"
    if t < 32:   return "☀️ Warm"
    return "🔥 Hot"


# ── APP LAYOUT ────────────────────────────────────────────────
st.markdown('<div class="main-header">🌤️ Delhi Weather Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ensemble ML model (XGBoost + LightGBM) '
            '· Trained on Delhi 2013–2017 daily data</div>',
            unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Models not found. Please run the training notebook first, "
             "then restart the app.")
    st.stop()

# ── Sidebar: Inputs ───────────────────────────────────────────
with st.sidebar:
    st.header("📋 Input Today's Weather")
    st.markdown("*Enter today's observed values:*")

    temp = st.slider("🌡️ Mean Temperature (°C)",
                     min_value=0.0, max_value=45.0, value=22.0, step=0.5)
    humidity = st.slider("💧 Humidity (%)",
                         min_value=10.0, max_value=100.0, value=65.0, step=1.0)
    wind_speed = st.slider("💨 Wind Speed (km/h)",
                           min_value=0.0, max_value=50.0, value=8.0, step=0.5)
    pressure = st.slider("🌀 Mean Pressure (hPa)",
                         min_value=990.0, max_value=1030.0, value=1010.0, step=0.5)

    st.markdown("---")
    st.subheader("📅 Recent History")
    temp_yesterday   = st.number_input("Yesterday's Temp (°C)",
                                       value=21.0, step=0.5)
    temp_2days_ago   = st.number_input("2 Days Ago Temp (°C)",
                                       value=20.0, step=0.5)

    predict_btn = st.button("🔮 Predict Tomorrow")

# ── Main Panel ────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    if predict_btn:
        xgb_pred, lgb_pred, ens_pred = predict_temperature(
            temp, humidity, wind_speed, pressure,
            temp_yesterday, temp_2days_ago
        )

        st.subheader("📊 Prediction Results")
        c1, c2, c3 = st.columns(3)

        for col_widget, model_name, pred in zip(
            [c1, c2, c3],
            ["XGBoost", "LightGBM", "🏆 Ensemble"],
            [xgb_pred, lgb_pred, ens_pred]
        ):
            with col_widget:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{model_name}</div>
                    <div class="metric-value">{pred:.1f}°C</div>
                    <div class="metric-label">{temp_to_emoji(pred)}</div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        models_names = ['XGBoost', 'LightGBM', 'Ensemble']
        preds        = [xgb_pred, lgb_pred, ens_pred]
        colors       = ['#2E6DA4', '#E05C5C', '#FFD700']
        bars = ax.bar(models_names, preds, color=colors, edgecolor='white',
                      width=0.5)
        ax.set_ylabel('Predicted Temp (°C)')
        ax.set_title('Model Predictions Comparison', fontsize=11)
        ax.set_ylim(min(preds) - 2, max(preds) + 2)
        for bar, val in zip(bars, preds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}°C', ha='center', va='bottom', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig)

    else:
        st.info("👈 Enter today's weather values in the sidebar and press **Predict Tomorrow**.")

with col2:
    st.subheader("ℹ️ Feature Guide")
    st.markdown("""
    <div class="info-box"><b>🌡️ Temperature</b><br>
    Mean daily air temperature in °C. Delhi ranges from ~6°C (Jan) to ~39°C (May-Jun).</div>
    <div class="info-box"><b>💧 Humidity</b><br>
    Relative humidity %. High in monsoon (Jul-Sep), low in summer/winter.</div>
    <div class="info-box"><b>💨 Wind Speed</b><br>
    Mean wind in km/h. Strong winds are common before monsoon or storms.</div>
    <div class="info-box"><b>🌀 Pressure</b><br>
    Atmospheric pressure in hPa. Normal = 1005–1020 hPa. Low pressure = rain likely.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Model Info")
    st.markdown("""
    | Metric | XGBoost | LightGBM |
    |--------|---------|----------|
    | RMSE   | ~2.1°C  | ~2.2°C   |
    | MAE    | ~1.5°C  | ~1.6°C   |
    | R²     | ~0.95   | ~0.94    |
    
    *Ensemble combines both using inverse-RMSE weighting.*
    """)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ | Weather Prediction ML Project | "
           "Dataset: Delhi 2013–2017 | Models: XGBoost + LightGBM + SHAP")
