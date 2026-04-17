"""
B3  Sequence-Aware Model
========================
Without PyTorch/TF we implement the "sequence model" concept by extracting
rich time-series features directly from the changelog trajectory for each deal.
These features capture dynamics the static flat-vector model cannot see:
  - probability slope / acceleration / momentum / drawdown
  - stage velocity and reversals
  - update cadence patterns (gaps, bursts, irregular pauses)
  - amount trend stability

A second XGBoost is trained with static + sequence features and compared
to the static-only baseline (from calibrated_model.py).

Outputs:
  outputs/25_sequence_feature_importance.png
  outputs/26_model_comparison_roc.png
  outputs/27_trajectory_risk_patterns.png
  outputs/sequence_open_scores.csv
"""

import sys, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from xgboost import XGBClassifier

from feature_engineering import (load_raw, engineer_features,
                                  ALL_FEATURES, NUMERIC_FEATURES, CAT_FEATURES)

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"
SEED    = 42
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")


# ── 1. EXTRACT SEQUENCE FEATURES FROM RAW CHANGELOG ───────────────────────
def extract_sequence_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each chance_id, compute trajectory-level statistics that capture
    the shape and dynamics of the deal lifecycle — not just the endpoint.
    """
    raw = raw.copy().sort_values(["chance_id", "updated_dt"]).reset_index(drop=True)
    rows = []

    for cid, grp in raw.groupby("chance_id", sort=False):
        n = len(grp)
        probs  = grp["probability"].values.astype(float)
        stages = grp["stage_en"].map({
            "Lead": 1, "Interest phase": 2, "Information Stage": 3,
            "Proposal stage": 4, "Negotiation phase": 5, "Verbal Confirmation": 6,
            "Order for registration - SC / COF": 7,
            "Order in revision - SC / COF": 8,
            "Order received - Sales": 9, "Draft": 2,
        }).fillna(3).values.astype(float)
        amounts = grp["amount"].values.astype(float)
        times   = (grp["updated_dt"] - grp["updated_dt"].iloc[0]).dt.days.values.astype(float)

        # ── Probability trajectory ─────────────────────────────────────
        if n >= 2:
            # Linear slope (prob per day)
            if times[-1] > 0:
                prob_slope = np.polyfit(times, probs, 1)[0]
            else:
                prob_slope = 0.0
            # Early vs late slope (acceleration)
            mid = max(1, n // 2)
            early_slope = (probs[mid] - probs[0])  / max(times[mid], 1)
            late_slope  = (probs[-1] - probs[mid]) / max(times[-1] - times[mid], 1)
            prob_acceleration = late_slope - early_slope
            # Momentum: last 20% vs first 20%
            seg = max(1, n // 5)
            prob_momentum = probs[-seg:].mean() - probs[:seg].mean()
            # Max drawdown from peak
            peak = np.maximum.accumulate(probs)
            prob_max_drawdown = (peak - probs).max()
            # Turning points (direction reversals)
            diff = np.diff(probs)
            sign_changes = np.sum(np.diff(np.sign(diff[diff != 0])) != 0) if len(diff[diff != 0]) > 1 else 0
            prob_turning_points = int(sign_changes)
            # Final direction: rising / falling / flat
            recent_diff = probs[-1] - probs[max(0, n-3)]
            prob_final_trend = 1 if recent_diff > 2 else (-1 if recent_diff < -2 else 0)
            # Volatility ratio: recent std / overall std
            overall_std = np.std(probs) + 1e-9
            recent_std  = np.std(probs[max(0, n-4):]) + 1e-9
            prob_volatility_ratio = recent_std / overall_std
        else:
            prob_slope = prob_acceleration = prob_momentum = 0.0
            prob_max_drawdown = 0.0
            prob_turning_points = 0
            prob_final_trend = 0
            prob_volatility_ratio = 1.0

        # ── Stage trajectory ───────────────────────────────────────────
        stage_range     = float(stages.max() - stages.min())
        stage_reversals = int(np.sum(np.diff(stages) < 0))
        if n >= 2 and times[-1] > 0:
            stage_velocity = (stages[-1] - stages[0]) / times[-1]
        else:
            stage_velocity = 0.0

        # ── Amount stability ───────────────────────────────────────────
        amt_cv    = float(np.std(amounts) / (np.mean(amounts) + 1e-9))
        amt_trend = float(amounts[-1] - amounts[0]) / max(float(amounts[0]), 1)

        # ── Update cadence ─────────────────────────────────────────────
        if n >= 3:
            gaps     = np.diff(times)
            max_gap  = float(gaps.max())
            gap_cv   = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
            # Burst: max updates in rolling 7-day window
            burst = max(
                int(np.sum((times >= t) & (times <= t + 7)))
                for t in times
            )
            # Recency: fraction of updates in last 25% of lifetime
            if times[-1] > 0:
                recent_frac = float(np.sum(times >= times[-1] * 0.75) / n)
            else:
                recent_frac = 1.0
        else:
            max_gap = float(times[-1]) if n > 1 else 0.0
            gap_cv = 0.0
            burst  = n
            recent_frac = 1.0

        rows.append({
            "chance_id":              cid,
            # probability dynamics
            "seq_prob_slope":         prob_slope,
            "seq_prob_acceleration":  prob_acceleration,
            "seq_prob_momentum":      prob_momentum,
            "seq_prob_max_drawdown":  prob_max_drawdown,
            "seq_prob_turning_pts":   prob_turning_points,
            "seq_prob_final_trend":   prob_final_trend,
            "seq_prob_vol_ratio":     prob_volatility_ratio,
            # stage dynamics
            "seq_stage_range":        stage_range,
            "seq_stage_reversals":    stage_reversals,
            "seq_stage_velocity":     stage_velocity,
            # amount stability
            "seq_amt_cv":             amt_cv,
            "seq_amt_trend":          amt_trend,
            # update cadence
            "seq_max_gap":            max_gap,
            "seq_gap_cv":             gap_cv,
            "seq_burst":              burst,
            "seq_recent_frac":        recent_frac,
        })

    return pd.DataFrame(rows)


# ── 2. LOAD & BUILD FEATURES ───────────────────────────────────────────────
print("Loading data and engineering features …")
raw = load_raw()
df  = engineer_features(raw)

for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))

print("Extracting sequence features …")
seq_feats = extract_sequence_features(raw)
print(f"  Sequence features extracted for {len(seq_feats):,} deals")

df = df.merge(seq_feats, on="chance_id", how="left")

SEQ_COLS   = [c for c in df.columns if c.startswith("seq_")]
STATIC_COLS = [f for f in ALL_FEATURES if f in df.columns]
FULL_COLS   = STATIC_COLS + SEQ_COLS

# Fill NaN in seq columns (deals with <3 updates get 0)
df[SEQ_COLS] = df[SEQ_COLS].fillna(0)

# ── 3. TRAIN / TEST SPLIT ──────────────────────────────────────────────────
closed     = df[df["is_closed"] == 1].copy()
open_      = df[df["status"] == "Open"].copy()
train_mask = closed["registered_dt"] < TRAIN_CUTOFF

X_tr_static = closed.loc[train_mask, STATIC_COLS].fillna(0)
X_te_static = closed.loc[~train_mask, STATIC_COLS].fillna(0)
X_tr_full   = closed.loc[train_mask, FULL_COLS].fillna(0)
X_te_full   = closed.loc[~train_mask, FULL_COLS].fillna(0)
y_tr        = closed.loc[train_mask, "target_won"]
y_te        = closed.loc[~train_mask, "target_won"]

print(f"  Train: {len(X_tr_static):,}  |  Test: {len(X_te_static):,}")

# ── 4. TRAIN BOTH MODELS ───────────────────────────────────────────────────
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1)

print("Training static-only model …")
m_static = XGBClassifier(**XGB_PARAMS)
m_static.fit(X_tr_static, y_tr)
auc_static = roc_auc_score(y_te, m_static.predict_proba(X_te_static)[:, 1])
print(f"  Static-only AUC : {auc_static:.4f}")

print("Training static + sequence model …")
m_full = XGBClassifier(**XGB_PARAMS)
m_full.fit(X_tr_full, y_tr)
auc_full = roc_auc_score(y_te, m_full.predict_proba(X_te_full)[:, 1])
print(f"  Static+Seq  AUC : {auc_full:.4f}  (Δ={auc_full-auc_static:+.4f})")

# ── 5. ROC COMPARISON CHART ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for model, X_te, label, color in [
    (m_static, X_te_static, f"Static features  (AUC={auc_static:.4f})", "#C53030"),
    (m_full,   X_te_full,   f"+ Sequence features (AUC={auc_full:.4f})", "#276749"),
]:
    fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
    ax.plot(fpr, tpr, linewidth=2, label=label, color=color)
ax.plot([0,1],[0,1],"k--",linewidth=0.8,label="Random")
ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
ax.set_title("B3 — ROC Curve: Static vs Static+Sequence Features\n"
             "Sequence features capture deal lifecycle dynamics", fontsize=11)
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "26_model_comparison_roc.png", dpi=150)
plt.close(fig)
print("  Saved 26_model_comparison_roc.png")

# ── 6. SEQUENCE FEATURE IMPORTANCE ────────────────────────────────────────
imp = pd.Series(m_full.feature_importances_, index=FULL_COLS).sort_values(ascending=False)
seq_imp    = imp[imp.index.str.startswith("seq_")]
static_imp = imp[~imp.index.str.startswith("seq_")]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
seq_imp.head(16).plot.barh(ax=ax, color="#2B6CB0")
ax.invert_yaxis()
ax.set_title("B3 — Sequence Feature Importance\n(gain in full model)")
ax.set_xlabel("Feature importance (gain)")

ax = axes[1]
static_imp.head(16).plot.barh(ax=ax, color="#276749")
ax.invert_yaxis()
ax.set_title("Top Static Features\n(for comparison)")
ax.set_xlabel("Feature importance (gain)")

fig.suptitle("B3 — What Sequence Features Add to the Model", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "25_sequence_feature_importance.png", dpi=150)
plt.close(fig)
print("  Saved 25_sequence_feature_importance.png")

# ── 7. TRAJECTORY RISK PATTERNS ───────────────────────────────────────────
# Show how trajectory shape relates to outcome for the most predictive seq features
top3_seq = seq_imp.head(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, feat in zip(axes, top3_seq):
    won  = closed.loc[closed["target_won"] == 1, feat].dropna()
    lost = closed.loc[closed["target_won"] == 0, feat].dropna()
    ax.hist(won.clip(*np.percentile(closed[feat].dropna(), [1,99])),
            bins=40, color="#276749", alpha=0.55, density=True, label="Won")
    ax.hist(lost.clip(*np.percentile(closed[feat].dropna(), [1,99])),
            bins=40, color="#C53030", alpha=0.55, density=True, label="Lost")
    ax.set_title(f"{feat}", fontsize=9)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

fig.suptitle("B3 — Top Sequence Features: Won vs Lost Distributions\n"
             "Separation = predictive power the static model misses", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "27_trajectory_risk_patterns.png", dpi=150)
plt.close(fig)
print("  Saved 27_trajectory_risk_patterns.png")

# ── 8. SCORE OPEN DEALS WITH SEQUENCE MODEL ───────────────────────────────
X_open_full = open_[FULL_COLS].fillna(0)
open_["seq_win_prob"] = m_full.predict_proba(X_open_full)[:, 1]

export = open_[["chance_id","business_unit","org_country","pricelist",
                "amount","deal_age","seq_win_prob"] + SEQ_COLS].copy()
export.to_csv(OUT_DIR / "sequence_open_scores.csv", index=False)
print("  Saved sequence_open_scores.csv")

# ── 9. SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n── B3 Summary ──")
print(f"  Sequence features extracted  : {len(SEQ_COLS)}")
print(f"  AUC lift from sequences       : {auc_full-auc_static:+.4f}")
print(f"  Top sequence predictors       : {top3_seq}")
print(f"  Most important seq feature    : {seq_imp.index[0]}  "
      f"(importance={seq_imp.iloc[0]:.4f})")
print("\nDone — sequence_features_model.py")
