"""Benchmark preprocessing and estimate full training time."""
import time, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.data_loader import load_data
from src.detection.fvg import detect_fvg
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from src.detection.orderblock import detect_order_blocks
from src.detection.structure import detect_bos, detect_choch
from src.features.builder import build_feature_pipeline
from src.features.labels import create_labels
from src.strategy.rule_based import generate_signals, signals_to_setups

with open('config/detection_config.yaml') as f:
    det_cfg = yaml.safe_load(f)['detection']
with open('config/risk_config.yaml') as f:
    risk_cfg = yaml.safe_load(f)['risk']
with open('config/model_config.yaml') as f:
    model_cfg = yaml.safe_load(f)['model']

SYMBOL = 'EURUSD'
print(f"=== Benchmarking {SYMBOL} M15 pipeline ===")

t0 = time.time()
candles = load_data(SYMBOL, 'M15')
t_data = time.time() - t0
print(f"[{t_data:.1f}s] Data: {len(candles):,} bars  {candles.index[0].date()} → {candles.index[-1].date()}")

t1 = time.time()
detections = {
    'fvg': detect_fvg(candles, min_gap_atr=det_cfg['fvg_min_gap_atr']),
    'order_blocks': detect_order_blocks(candles, min_move_atr=det_cfg['order_block_min_move_atr'], lookback=det_cfg['ob_lookback']),
    'liquidity_sweeps': detect_liquidity_sweeps(candles, lookback=det_cfg['liquidity_lookback'], threshold_atr=det_cfg.get('liquidity_sweep_atr_multiplier', 0.5), swing_lookback=det_cfg['swing_lookback']),
    'equal_levels': detect_equal_highs_lows(candles, swing_lookback=det_cfg['swing_lookback']),
    'bos': detect_bos(candles, confirmation_bars=det_cfg['bos_confirmation_bars'], swing_lookback=det_cfg['swing_lookback']),
    'choch': detect_choch(candles, swing_lookback=det_cfg['choch_swing_lookback']),
}
t_det = time.time() - t1
print(f"[{t_det:.1f}s] Detection: FVGs={len(detections['fvg'])}  OBs={len(detections['order_blocks'])}  Sweeps={len(detections['liquidity_sweeps'])}")

t2 = time.time()
signals = generate_signals(candles, detections, risk_cfg)
setups  = signals_to_setups(signals)
t_sig = time.time() - t2
print(f"[{t_sig:.1f}s] Signals: {len(signals)}  Setups: {len(setups)}")

t3 = time.time()
labels_df = create_labels(setups, candles, max_holding_bars=12)
t_lbl = time.time() - t3
pos_rate = labels_df['binary'].mean() if not labels_df.empty else 0
print(f"[{t_lbl:.1f}s] Labels: {len(labels_df)} samples  pos_rate={pos_rate:.3f}")

t4 = time.time()
feats_full = build_feature_pipeline(candles, detections, normalize=True)
feats = feats_full.iloc[labels_df['index'].values]
feats.index = candles.index[labels_df['index'].values]
t_feat = time.time() - t4
print(f"[{t_feat:.1f}s] Features: {feats.shape}  (47 features)")

pre_train_total = time.time() - t0
print(f"\n--- Pre-training total: {pre_train_total:.1f}s ---")

# Walk-forward folds estimate
train_months = model_cfg.get('walk_forward', {}).get('train_months', 24)
val_months   = model_cfg.get('walk_forward', {}).get('val_months', 2)
test_months  = model_cfg.get('walk_forward', {}).get('test_months', 1)
import pandas as pd
ts = feats.index
splits = []
from dateutil.relativedelta import relativedelta
cur = ts.min()
while True:
    te = cur + pd.DateOffset(months=train_months)
    ve = te  + pd.DateOffset(months=val_months)
    xe = ve  + pd.DateOffset(months=test_months)
    if xe > ts.max(): break
    splits.append((cur, te, ve, xe))
    cur = cur + pd.DateOffset(months=test_months)
n_folds = len(splits)
print(f"Walk-forward folds: {n_folds}  (train={train_months}m val={val_months}m test={test_months}m)")

# Estimate model training time
# LightGBM ~800 trees on ~1000 samples: ~8s
# XGBoost ~800 trees: ~10s
# CatBoost ~800 trees: ~12s
# LSTM (skipped if TF slow): ~60s
# Meta fit: <1s
# Bayesian opt: 30 iter × 200 rounds = adds ~30s IF enabled
lgb_s   = 8
xgb_s   = 10
cat_s   = 12
lstm_s  = 60   # conservative; TF compile overhead
meta_s  = 1
per_fold = lgb_s + xgb_s + cat_s + lstm_s + meta_s
print(f"\nEstimated time per fold (LGB+XGB+CAT+LSTM+meta): {per_fold}s")
total_train_s = per_fold * n_folds
print(f"Estimated training time for {SYMBOL}: {total_train_s}s = {total_train_s/60:.1f} min")
print(f"  (includes {n_folds} folds × {per_fold}s)")

N_SYMBOLS = 13
with_preproc = pre_train_total + total_train_s
print(f"\n=== FULL UNIVERSE ESTIMATE ===")
print(f"Per symbol (sequential): {with_preproc:.0f}s = {with_preproc/60:.1f} min")
print(f"13 symbols sequential:   {with_preproc*N_SYMBOLS/60:.0f} min = {with_preproc*N_SYMBOLS/3600:.1f} hr")

import os
cpus = os.cpu_count()
print(f"\nCPUs available: {cpus}")
parallelism = min(cpus, N_SYMBOLS)
parallel_time = (with_preproc * N_SYMBOLS) / parallelism
print(f"With {parallelism}-way parallelism: {parallel_time/60:.0f} min = {parallel_time/3600:.1f} hr")
