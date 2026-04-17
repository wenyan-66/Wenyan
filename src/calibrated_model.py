"""
A1  Probability Calibration       — isotonic regression on XGBoost scores
A2  Coverage Extension            — train on unfiltered data (all 38k deals)
A4  Bootstrap Forecast Intervals  — 90% CI on monthly expected revenue
B4  SHAP Per-Deal Explanations    — global summary + per-deal top drivers

Outputs:
  outputs/13_calibration_curve.png
  outputs/14_shap_summary.png
  outputs/15_bootstrap_forecast.png
  outputs/calibrated_open_scores.csv
"""

import sys, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from feature_engineering import load_raw, engineer_features, ALL_FEATURES, NUMERIC_FEATURES, CAT_FEATURES

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
SEED    = 42
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")

# ── 1. LOAD & ENGINEER ─────────────────────────────────────────────────────
print("Loading and engineering features …")
raw = load_raw()
df  = engineer_features(raw)
print(f"  Total opportunities: {len(df):,}  |  "
      f"Won: {(df.status=='Won').sum():,}  "
      f"Lost: {(df.status=='Lost').sum():,}  "
      f"Open: {(df.status=='Open').sum():,}")

# Encode categoricals
for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))

# ── 2. SPLIT — A2 FULL COVERAGE (no filters) ──────────────────────────────
# Train on ALL closed deals (no intercompany filter — A2 improvement)
closed = df[df["is_closed"] == 1].copy()
open_  = df[df["status"] == "Open"].copy()

feats  = [f for f in ALL_FEATURES if f in df.columns]

X_closed = closed[feats].fillna(0)
y_closed = closed["target_won"]

# Temporal train/test split
train_mask = closed["registered_dt"] < TRAIN_CUTOFF
X_train, X_test = X_closed[train_mask], X_closed[~train_mask]
y_train, y_test = y_closed[train_mask], y_closed[~train_mask]
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Open: {len(open_):,}")

# ── 3. BASE XGBoost ────────────────────────────────────────────────────────
print("Training base XGBoost …")
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1,
)
xgb.fit(X_train, y_train)
auc_base  = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
brier_base = brier_score_loss(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"  Base XGB — AUC: {auc_base:.4f}  Brier: {brier_base:.4f}")

# ── 4. CALIBRATION (A1) ────────────────────────────────────────────────────
print("Calibrating probabilities (isotonic) …")
cal_model = CalibratedClassifierCV(xgb, method="isotonic")
cal_model.fit(X_test, y_test)   # fit calibration on held-out set

prob_raw = xgb.predict_proba(X_test)[:, 1]
prob_cal = cal_model.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, prob_cal)
print(f"  Calibrated   — Brier: {brier_cal:.4f}  (Δ={brier_base-brier_cal:+.4f})")

# Calibration curve
fig, ax = plt.subplots(figsize=(7, 6))
for probs, label, color in [(prob_raw, "Uncalibrated XGB", "#C53030"),
                             (prob_cal, "Isotonic calibrated", "#276749")]:
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label=label, color=color)
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (actual win rate)")
ax.set_title("A1 — Probability Calibration Curve\n"
             "Closer to diagonal = more trustworthy probabilities", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "13_calibration_curve.png", dpi=150)
plt.close(fig)
print("  Saved 13_calibration_curve.png")

# ── 5. SHAP EXPLANATIONS (B4) ─────────────────────────────────────────────
print("Computing SHAP values …")
explainer   = shap.TreeExplainer(xgb)
shap_test   = explainer(X_test)

# Global summary bar
fig, ax = plt.subplots(figsize=(9, 7))
shap.plots.bar(shap_test, max_display=20, ax=ax, show=False)
ax.set_title("B4 — SHAP Global Feature Importance\n"
             "(mean |SHAP value| across test set)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "14a_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Beeswarm (impact direction) — shap manages its own figure
shap.plots.beeswarm(shap_test, max_display=15, show=False, plot_size=(9, 7))
plt.title("B4 — SHAP Beeswarm: Feature Impact on Win-Probability\n"
          "(red = high feature value; each dot = one deal)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "14b_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("  Saved 14a_shap_bar.png  14b_shap_beeswarm.png")

# ── 6. BOOTSTRAP FORECAST INTERVALS (A4) ──────────────────────────────────
print("Bootstrap forecast intervals (50 resamples) …")
X_open = open_[feats].fillna(0)
X_all_train = X_closed[train_mask]
y_all_train = y_closed[train_mask]

N_BOOT = 50
boot_preds = np.zeros((N_BOOT, len(X_open)))

for i in range(N_BOOT):
    rng = np.random.default_rng(SEED + i)
    idx = rng.integers(0, len(X_all_train), size=len(X_all_train))
    xb  = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=int(SEED+i), verbosity=0, n_jobs=-1,
    )
    xb.fit(X_all_train.iloc[idx], y_all_train.iloc[idx])
    boot_preds[i] = xb.predict_proba(X_open)[:, 1]
    if (i + 1) % 10 == 0:
        print(f"    bootstrap {i+1}/{N_BOOT}")

# Point estimate from calibrated model
open_["pred_win_prob_cal"] = cal_model.predict_proba(X_open)[:, 1]
open_["pred_win_prob_p05"] = np.percentile(boot_preds, 5,  axis=0)
open_["pred_win_prob_p95"] = np.percentile(boot_preds, 95, axis=0)
open_["expected_rev"]      = open_["amount"] * open_["pred_win_prob_cal"]
open_["rev_p05"]           = open_["amount"] * open_["pred_win_prob_p05"]
open_["rev_p95"]           = open_["amount"] * open_["pred_win_prob_p95"]

# Monthly forecast aggregation
open_["close_month"] = open_["saledate"].dt.to_period("M")
monthly = (open_.groupby("close_month")
           .agg(expected_rev=("expected_rev", "sum"),
                rev_p05     =("rev_p05",      "sum"),
                rev_p95     =("rev_p95",      "sum"),
                n_deals     =("chance_id",    "count"))
           .reset_index())
monthly["close_month"] = monthly["close_month"].astype(str)

# Limit to next 18 months
now = pd.Timestamp("today").to_period("M").to_timestamp()
future = monthly[pd.to_datetime(monthly["close_month"]) >= now].head(18)

if len(future) > 0:
    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(future))
    ax.fill_between(x, future["rev_p05"]/1e6, future["rev_p95"]/1e6,
                    alpha=0.25, color="#2B6CB0", label="90% CI")
    ax.plot(x, future["expected_rev"]/1e6, "o-", color="#2B6CB0",
            linewidth=2, label="Expected revenue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(future["close_month"], rotation=45, ha="right")
    ax.set_ylabel("Revenue (M)")
    ax.set_title("A4 — Monthly Forecast with 90% Bootstrap Confidence Interval\n"
                 "(open deals only, grouped by CRM saledate)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "15_bootstrap_forecast.png", dpi=150)
    plt.close(fig)
    print("  Saved 15_bootstrap_forecast.png")

# ── 7. SHAP PER OPEN DEAL (top 5 drivers) ─────────────────────────────────
print("Computing SHAP for open deals …")
shap_open   = explainer(X_open)
shap_vals   = shap_open.values          # (n_open, n_feats)
top_shap_idx = np.argsort(np.abs(shap_vals), axis=1)[:, -5:][:, ::-1]

shap_rows = []
for i, cid in enumerate(open_["chance_id"].values):
    for rank, fi in enumerate(top_shap_idx[i]):
        shap_rows.append({
            "chance_id":   cid,
            "shap_rank":   rank + 1,
            "feature":     feats[fi],
            "shap_value":  shap_vals[i, fi],
            "feature_val": float(X_open.iloc[i, fi]),
        })
pd.DataFrame(shap_rows).to_csv(OUT_DIR / "shap_open_deals.csv", index=False)
print("  Saved shap_open_deals.csv")

# ── 8. EXPORT CALIBRATED SCORES ────────────────────────────────────────────
export_cols = [
    "chance_id", "business_unit", "org_country", "pricelist",
    "status", "amount", "probability", "deal_age",
    "pred_win_prob_cal", "pred_win_prob_p05", "pred_win_prob_p95",
    "expected_rev", "rev_p05", "rev_p95",
]
avail = [c for c in export_cols if c in open_.columns]
open_[avail].to_csv(OUT_DIR / "calibrated_open_scores.csv", index=False)
print("  Saved calibrated_open_scores.csv")

# ── 9. COVERAGE COMPARISON ────────────────────────────────────────────────
orig_scored = 1734   # from forecast_won_lost.parquet
new_scored  = len(open_)
print(f"\n── Coverage improvement (A2) ──")
print(f"  Original RF model (filtered): {orig_scored:,} open deals scored")
print(f"  New XGB model (unfiltered)  : {new_scored:,} open deals scored")
print(f"  Coverage gain               : +{new_scored-orig_scored:,} deals "
      f"({(new_scored/orig_scored-1)*100:.0f}% more)")

print("\nDone — calibrated_model.py")
