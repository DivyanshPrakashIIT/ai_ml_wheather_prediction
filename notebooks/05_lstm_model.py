
# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10: LSTM MODEL — Bidirectional LSTM + Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════════
#
# PIPELINE    : LSTM (independent — uses its own feature engineering)
# ARCHITECTURE: BiLSTM(256) → BiLSTM(128) + Residual → MultiHeadAttention
#               + LSTM(64) → Merge → Dense(128→64→32→1)
#
# DATA FLOW:
#   Input  : data/raw/Train.csv + data/raw/Test.csv
#   Builds : its own features (lags, rolling, interactions) on top of raw data
#   Output : data/predictions/lstm.csv  (columns: date, id, prediction)
#            models/lstm_model.keras
#
# ALIGNMENT:  114 test-date predictions matching data/raw/Test.csv dates.
#             LOOKBACK=30 sliding window — test windows use the last 30 train
#             rows as context, so all 114 test dates are covered.
#
# WHY SEPARATE PREPROCESSING?
#   LSTM needs: MinMaxScaler (for gradient stability), sliding windows (3D
#   input), specific lag/rolling features suited to sequence models.
#   Mixing with the XGB/LGB pipeline would corrupt both models.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts/Colab
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
    Input, Add, Concatenate, GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

# ── CELL 1: CONFIG ────────────────────────────────────────────────────────────
# Paths — relative to project root (run from weather-prediction-ml-main/)
TRAIN_PATH  = 'data/raw/Train.csv'
TEST_PATH   = 'data/raw/Test.csv'
DATE_COL    = 'date'
TARGET_COL  = 'meantemp'
LOOKBACK    = 30       # days of context per prediction
EPOCHS      = 250      # EarlyStopping will stop well before this
BATCH_SIZE  = 8
VAL_SPLIT   = 0.15     # fraction of train windows used for validation

print("=" * 62)
print("  LSTM MODEL — Bidirectional LSTM + MultiHeadAttention")
print("=" * 62)

# ── CELL 2: LOAD RAW DATA ─────────────────────────────────────────────────────
# LSTM reads from raw CSVs and builds its own features.
# This keeps the LSTM pipeline fully independent of the XGB/LGB pipeline.

train_df = pd.read_csv(TRAIN_PATH, parse_dates=[DATE_COL])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=[DATE_COL])

train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
test_df  = test_df.sort_values(DATE_COL).reset_index(drop=True)

# Remove duplicate dates in test (2017-01-01 appears in both train and test)
dupes = test_df.duplicated(subset=DATE_COL, keep=False).sum()
if dupes > 0:
    print(f'  Duplicate dates in test: {dupes} — keeping last')
    test_df = test_df.drop_duplicates(subset=DATE_COL, keep='last').reset_index(drop=True)

print(f"\n  Train : {len(train_df)} rows  "
      f"{train_df[DATE_COL].iloc[0].date()} -> {train_df[DATE_COL].iloc[-1].date()}")
print(f"  Test  : {len(test_df)} rows   "
      f"{test_df[DATE_COL].iloc[0].date()} -> {test_df[DATE_COL].iloc[-1].date()}")

# ── CELL 3: LSTM-SPECIFIC FEATURE ENGINEERING ─────────────────────────────────
#
# WHY LSTM NEEDS ITS OWN FEATURES?
# ──────────────────────────────────
# XGB/LGB use lag features as standalone columns (each lag = independent input).
# LSTM sees sequences — it learns temporal patterns from the sliding window
# itself, but additional derived features (velocity, momentum, z-scores) help
# it distinguish seasonal signals from random noise more quickly.
#
# These features are applied BEFORE scaling and windowing, so they become
# additional channels in the 3D input tensor (samples, lookback, features).

def engineer_features(df):
    """
    Build LSTM-optimised time-series features on top of raw weather columns.
    Applied identically to train and test — no target leakage.
    """
    df = df.copy()

    # -- Velocity (1st differences) ------------------------------------------
    # How fast is temperature changing? Helps LSTM detect onset of heatwaves.
    df['temp_diff_1']  = df['meantemp'].diff(1)
    df['temp_diff_2']  = df['meantemp'].diff(2)
    df['temp_diff_3']  = df['meantemp'].diff(3)
    df['temp_diff_7']  = df['meantemp'].diff(7)
    df['temp_diff_14'] = df['meantemp'].diff(14)

    # -- Acceleration (2nd derivative) ---------------------------------------
    # Is the rate of temperature change itself accelerating?
    df['temp_accel'] = df['temp_diff_1'].diff(1)

    # -- Rolling statistics (pre-computed for LSTM input) --------------------
    # These are also computed inside the window, but providing them
    # as explicit features speeds up convergence.
    df['temp_roll_mean_7']  = df['meantemp'].rolling(7).mean()
    df['temp_roll_mean_14'] = df['meantemp'].rolling(14).mean()
    df['temp_roll_std_7']   = df['meantemp'].rolling(7).std()
    df['temp_roll_std_14']  = df['meantemp'].rolling(14).std()
    df['temp_roll_min_7']   = df['meantemp'].rolling(7).min()
    df['temp_roll_max_7']   = df['meantemp'].rolling(7).max()
    df['temp_roll_range_7'] = df['temp_roll_max_7'] - df['temp_roll_min_7']

    # -- Momentum (short vs long trend) --------------------------------------
    df['temp_momentum_7']  = df['temp_roll_mean_7']  - df['temp_roll_mean_14']
    df['temp_momentum_14'] = df['temp_roll_mean_14'] - df['meantemp'].rolling(21).mean()

    # -- Exponential moving averages -----------------------------------------
    # EMA reacts faster than simple rolling mean — captures recent regime shifts
    df['temp_ema_7']    = df['meantemp'].ewm(span=7,  adjust=False).mean()
    df['temp_ema_14']   = df['meantemp'].ewm(span=14, adjust=False).mean()
    df['temp_ema_diff'] = df['temp_ema_7'] - df['temp_ema_14']

    # -- Z-score relative to 30-day window -----------------------------------
    # How extreme is today's temperature relative to the past month?
    roll_mean_30       = df['meantemp'].rolling(30).mean()
    roll_std_30        = df['meantemp'].rolling(30).std().replace(0, 1)
    df['temp_zscore_30'] = (df['meantemp'] - roll_mean_30) / roll_std_30

    # -- Humidity derived features -------------------------------------------
    df['humidity_diff_1']    = df['humidity'].diff(1)
    df['humidity_roll_7']    = df['humidity'].rolling(7).mean()
    df['humidity_roll_std_7']= df['humidity'].rolling(7).std()

    # -- Wind derived --------------------------------------------------------
    df['wind_roll_7']  = df['wind_speed'].rolling(7).mean()
    df['wind_diff_1']  = df['wind_speed'].diff(1)

    # -- Pressure derived ----------------------------------------------------
    df['pressure_diff_1'] = df['meanpressure'].diff(1)
    df['pressure_diff_3'] = df['meanpressure'].diff(3)
    df['pressure_roll_7'] = df['meanpressure'].rolling(7).mean()

    # -- Cross-feature interactions ------------------------------------------
    df['humidity_x_wind']  = df['humidity']     * df['wind_speed']
    df['temp_x_humidity']  = df['meantemp']      * df['humidity']
    df['temp_x_wind']      = df['meantemp']      * df['wind_speed']
    df['wind_x_pressure']  = df['wind_speed']    * df['meanpressure']

    # Fill NaN from diff/rolling — forward-fill then zero for leading rows
    df = df.ffill().fillna(0)

    return df


train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

print(f"\n  Features after engineering : {len(train_df.columns)} columns")

# ── CELL 4: SCALE & WINDOW ────────────────────────────────────────────────────
#
# Scaling: MinMaxScaler fitted ONLY on train — no future leakage.
# Target (meantemp) is placed at column index 0 so inverse_transform is easy.
# Windowing: combine train+test for continuous windows; split by index.

feature_cols = [c for c in train_df.columns if c != DATE_COL]
# Ensure target is column 0 (required for clean inverse transform)
feature_cols = [TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]
n_features   = len(feature_cols)

print(f"  Total input features       : {n_features}")

# Combine for continuous sliding windows across train/test boundary
combined_df     = pd.concat([train_df, test_df], ignore_index=True)\
                    .sort_values(DATE_COL).reset_index(drop=True)

scaler          = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[feature_cols])             # fit on train only
combined_scaled = scaler.transform(combined_df[feature_cols])

# Build sliding windows: X shape = (samples, LOOKBACK, features), y = next step target
def create_sliding_window(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])   # col 0 = meantemp
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = create_sliding_window(combined_scaled, LOOKBACK)

# Split index: first test window's target = first test row
split_idx        = len(train_df) - LOOKBACK
X_train_w, y_train_w = X_all[:split_idx], y_all[:split_idx]
X_test_w,  y_test_w  = X_all[split_idx:], y_all[split_idx:]

print(f"  X_train windows            : {X_train_w.shape}")
print(f"  X_test  windows            : {X_test_w.shape}  "
      f"(should match {len(test_df)} test rows)")

# ── CELL 5: BUILD MODEL ───────────────────────────────────────────────────────
#
# Architecture rationale:
#   BiLSTM(256): bidirectional — sees past AND near-future context in the window
#   BiLSTM(128) + residual: deeper representation, skip connection avoids vanishing grad
#   MultiHeadAttention(4 heads): learns which of the 30 days matter most per prediction
#   LSTM(64): sequential state complement to attention's global pooling
#   Dense head: 128→64→32→1, small dropout at each layer

n_lookback = X_train_w.shape[1]

inputs = Input(shape=(n_lookback, n_features), name='input')

# BiLSTM Block 1
x = Bidirectional(LSTM(256, return_sequences=True), name='bilstm_1')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

# BiLSTM Block 2 + Residual
x2 = Bidirectional(LSTM(128, return_sequences=True), name='bilstm_2')(x)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.10)(x2)

# Project Block-1 to match Block-2 dims for residual add (512 -> 256)
skip = Dense(256, use_bias=False, name='skip_proj')(x)
x    = Add(name='residual_add')([x2, skip])
x    = LayerNormalization(name='layer_norm')(x)

# Branch 1: Multi-Head Self-Attention
attn_out = MultiHeadAttention(
    num_heads=4, key_dim=64, dropout=0.1, name='mha'
)(x, x)
context = GlobalAveragePooling1D(name='attn_pool')(attn_out)   # (batch, 256)

# Branch 2: Sequential LSTM state
seq_out = LSTM(64, return_sequences=False, name='lstm_final')(x)
seq_out = Dropout(0.10)(seq_out)

# Merge both branches
merged = Concatenate(name='merge')([seq_out, context])         # (batch, 320)

# Dense prediction head
z = Dense(128, activation='relu', name='dense_1')(merged)
z = Dropout(0.10)(z)
z = Dense(64,  activation='relu', name='dense_2')(z)
z = Dense(32,  activation='relu', name='dense_3')(z)
outputs = Dense(1, name='output')(z)

model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_MHA_v3')

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer = optimizer,
    loss      = tf.keras.losses.Huber(delta=1.0),  # robust to outliers
    metrics   = ['mae']
)

print(f"\n  Model parameters : {model.count_params():,}")

# ── CELL 6: TRAIN ─────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=35,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=10,
        min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        filepath='models/lstm_best.keras',
        monitor='val_loss', save_best_only=True, verbose=0
    )
]

print(f"\n  Training (up to {EPOCHS} epochs, early stopping at patience=35)...")
history = model.fit(
    X_train_w, y_train_w,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_split= VAL_SPLIT,
    callbacks       = callbacks,
    verbose         = 1
)

best_epoch   = int(np.argmin(history.history['val_loss']))
best_val_loss = min(history.history['val_loss'])
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Best val_loss : {best_val_loss:.4f}")

# ── CELL 7: INVERSE TRANSFORM HELPER ─────────────────────────────────────────
def inverse_target(scaled_values, scaler, n_feat):
    """
    Inverse-transform ONLY the target column (index 0 = meantemp).
    Fills other columns with zeros as dummy placeholders.
    """
    dummy       = np.zeros((len(scaled_values), n_feat))
    dummy[:, 0] = scaled_values
    return scaler.inverse_transform(dummy)[:, 0]

# ── CELL 8: PREDICT & EVALUATE ────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test_w, verbose=0).flatten()
y_pred_c      = inverse_target(y_pred_scaled, scaler, n_features)
y_test_c      = inverse_target(y_test_w,      scaler, n_features)

# Train predictions (overfitting check)
y_train_pred_s = model.predict(X_train_w, verbose=0).flatten()
y_train_pred_c = inverse_target(y_train_pred_s, scaler, n_features)
y_train_c      = inverse_target(y_train_w,      scaler, n_features)

# Metrics
def compute_metrics(actual, pred, label):
    rmse  = np.sqrt(mean_squared_error(actual, pred))
    mae   = mean_absolute_error(actual, pred)
    r2    = r2_score(actual, pred)
    smape = np.mean(2 * np.abs(pred - actual) /
                    (np.abs(actual) + np.abs(pred) + 1e-8)) * 100
    print(f"  {label}")
    print(f"    RMSE  : {rmse:.4f} °C")
    print(f"    MAE   : {mae:.4f} °C")
    print(f"    R²    : {r2:.4f}")
    print(f"    sMAPE : {smape:.2f}%")
    return rmse, mae, r2

print("\n" + "=" * 62)
print("  LSTM TEST SET METRICS")
print("=" * 62)
rmse_test, mae_test, r2_test = compute_metrics(y_test_c,  y_pred_c,      "Test")
rmse_train, mae_train, r2_train = compute_metrics(y_train_c, y_train_pred_c, "Train (overfit check)")

# ── CELL 9: PREDICTION PLOT ───────────────────────────────────────────────────
# Get the test dates aligned to windowed predictions
test_dates_aligned = combined_df[DATE_COL].values[
    len(train_df) : len(train_df) + len(X_test_w)
]

os.makedirs('reports', exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates_aligned, y_test_c,  label='Actual',    color='steelblue',  lw=1.8)
ax.plot(test_dates_aligned, y_pred_c,  label='LSTM Pred', color='tomato',     lw=1.8, ls='--')
ax.set_title(f'LSTM — Actual vs Predicted  (RMSE={rmse_test:.3f}°C  R²={r2_test:.4f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved -> reports/lstm_actual_vs_predicted.png")

# Loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Train Loss', color='steelblue', lw=1.8)
ax.plot(history.history['val_loss'], label='Val Loss',   color='tomato',    lw=1.8, ls='--')
ax.axvline(best_epoch, color='green', ls=':', lw=1.5, label=f'Best epoch ({best_epoch})')
ax.set_title('LSTM Training Loss Curve', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss')
ax.legend()
plt.tight_layout()
plt.savefig('reports/lstm_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved -> reports/lstm_loss_curve.png")

# ── CELL 10: SAVE PREDICTIONS FOR ENSEMBLE ────────────────────────────────────
#
# ALIGNMENT GUARANTEE:
#   test_dates_aligned has exactly len(test_df) dates (114 rows),
#   corresponding to the 114 test-set dates in data/raw/Test.csv.
#   id = integer 0..113 for deterministic merge in 06_ensemble.py.

os.makedirs('data/predictions', exist_ok=True)

pred_df = pd.DataFrame({
    'date'      : test_dates_aligned,
    'id'        : range(len(y_pred_c)),
    'prediction': y_pred_c,
    'actual'    : y_test_c
})

pred_df.to_csv('data/predictions/lstm.csv', index=False)
print(f"\n  Saved -> data/predictions/lstm.csv  ({len(pred_df)} rows)")
print(f"  Date range: {pred_df.date.iloc[0]} -> {pred_df.date.iloc[-1]}")

# ── CELL 11: SAVE MODEL ───────────────────────────────────────────────────────
model.save('models/lstm_model.keras')
print("  Saved -> models/lstm_model.keras")

# ── CELL 12: SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  LSTM PIPELINE COMPLETE")
print("=" * 62)
print(f"  Architecture : BiLSTM(256) + BiLSTM(128) + MHA + LSTM(64)")
print(f"  Lookback     : {LOOKBACK} days")
print(f"  Features     : {n_features}")
print(f"  Epochs run   : {len(history.history['loss'])}")
print(f"  Best epoch   : {best_epoch}")
print(f"  Test RMSE    : {rmse_test:.4f} °C")
print(f"  Test R²      : {r2_test:.4f}")
print(f"  Predictions  : data/predictions/lstm.csv  ({len(pred_df)} rows)")
print("=" * 62)
print("\n  Next step -> run 05_arima_model.py, then 06_ensemble.py")
