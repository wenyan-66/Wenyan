"""
C4  Next-Best-Action Recommender
=================================
For each open opportunity, recommend the single highest-impact action
a sales rep should take next.

Approach (SHAP-driven):
  1. Load the calibrated XGBoost model, refit on full training data.
  2. Compute SHAP values for every open deal.
  3. For each deal, the top negative SHAP contributor = the feature most
     suppressing the win probability. Map that feature to an action.
  4. Validate historically: for each action proxy observable in closed
     deals, compare win rates of deals where the action occurred vs not.

Action map (feature → recommended action → business meaning):
  prob_change_last_update < 0  → "Review value proposition"
  days_since_last_update > 45  → "Re-engage customer now"
  zombie_deal = 1              → "Force close or disqualify"
  total_days_pushed > 90       → "Set hard deadline / stop extensions"
  rep_closing_rate (low)       → "Add sales support or re-assign"
  stage_rank (low) + prob>50   → "Push to advance stage"
  seq_prob_max_drawdown (high) → "Stabilise negotiation / address objections"
  seq_prob_final_trend < 0     → "Executive sponsor intervention"
  customer_win_ratio (low)     → "Customise approach for this account"
  default                      → "Schedule next customer touchpoint"

Outputs:
  outputs/28_nba_action_distribution.png
  outputs/29_nba_historical_validation.png
  outputs/open_deal_recommendations.csv
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
import shap
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from feature_engineering import (load_raw, engineer_features,
                                  ALL_FEATURES, NUMERIC_FEATURES, CAT_FEATURES)
from sequence_features_model import extract_sequence_features

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"
SEED    = 42
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")

# ── Action map: (feature_name, condition_lambda, label, description) ──────
# Priority order matters — first matching rule wins
ACTION_RULES = [
    ("zombie_deal",
     lambda v: v >= 1,
     "Force close or disqualify",
     "Deal inactive >1yr with no updates. Consuming pipeline bandwidth with no chance of closing."),

    ("total_days_pushed",
     lambda v: v > 90,
     "Set hard close deadline",
     "Saledate pushed >90 days total. Stop extensions — force a yes/no customer decision."),

    ("days_since_last_update",
     lambda v: v > 45,
     "Re-engage customer now",
     "No customer interaction in 45+ days. Momentum is lost — immediate outreach needed."),

    ("seq_prob_final_trend",
     lambda v: v < -5,
     "Executive sponsor intervention",
     "Probability declining in recent updates. Senior escalation can reset the deal trajectory."),

    ("seq_prob_max_drawdown",
     lambda v: v > 30,
     "Address objections / stabilise",
     "Large probability drop from peak. Unresolved objection is pulling the deal back."),

    ("prob_change_last_update",
     lambda v: v < -10,
     "Review value proposition",
     "Last update reduced probability significantly. Customer concerns need to be addressed directly."),

    ("rep_closing_rate",
     lambda v: v < 0.35,
     "Add sales support or re-assign",
     "Rep historical win rate is below average. Pairing with a senior rep or specialist may help."),

    ("stage_rank",
     lambda v: v <= 3,
     "Advance to next stage",
     "Deal stuck in early stage. Push for a formal proposal or demonstration to progress."),

    ("customer_win_ratio",
     lambda v: v < 0.25,
     "Customise account approach",
     "Low historical win rate at this account. Standard playbook is not working — try a different angle."),

    ("seq_amt_cv",
     lambda v: v > 0.5,
     "Stabilise deal scope",
     "Amount fluctuating heavily. Unclear scope is blocking commitment — define clear deliverables."),
]
DEFAULT_ACTION = ("Schedule customer touchpoint",
                  "No critical risk signals. Maintain momentum with a scheduled check-in.")


def recommend_action(row: pd.Series, shap_vals: np.ndarray,
                     feat_names: list) -> tuple:
    """
    For a single deal, identify the highest-urgency action.
    Priority: SHAP-magnitude-weighted rule matching.
    """
    # Build {feature: shap_value} dict (negative SHAP = suppressing win prob)
    shap_dict = dict(zip(feat_names, shap_vals))

    # Find features with the most negative SHAP (biggest drags on win prob)
    sorted_drags = sorted(
        [(f, sv) for f, sv in shap_dict.items() if sv < 0],
        key=lambda x: x[1]
    )
    drag_feats = [f for f, _ in sorted_drags]

    # Try rules in drag-feature order first, then in default priority order
    for check_order in [drag_feats, [r[0] for r in ACTION_RULES]]:
        for feat in check_order:
            for rule_feat, condition, label, desc in ACTION_RULES:
                if feat == rule_feat and feat in row.index:
                    try:
                        if condition(row[feat]):
                            return label, desc
                    except Exception:
                        pass

    return DEFAULT_ACTION


# ── 1. LOAD & BUILD FEATURES ───────────────────────────────────────────────
print("Loading data …")
raw = load_raw()
df  = engineer_features(raw)
seq = extract_sequence_features(raw)
df  = df.merge(seq, on="chance_id", how="left")

SEQ_COLS    = [c for c in df.columns if c.startswith("seq_")]
FULL_COLS   = [f for f in ALL_FEATURES if f in df.columns] + SEQ_COLS
df[SEQ_COLS] = df[SEQ_COLS].fillna(0)

for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))

# ── 2. TRAIN CALIBRATED MODEL ON ALL CLOSED DATA ──────────────────────────
print("Training calibrated model …")
closed = df[df["is_closed"] == 1].copy()
open_  = df[df["status"] == "Open"].copy()

X_cl = closed[FULL_COLS].fillna(0)
y_cl = closed["target_won"]

train_mask = closed["registered_dt"] < TRAIN_CUTOFF
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1)
xgb.fit(X_cl[train_mask], y_cl[train_mask])

cal = CalibratedClassifierCV(xgb, method="isotonic")
cal.fit(X_cl[~train_mask], y_cl[~train_mask])

# ── 3. SHAP FOR OPEN DEALS ─────────────────────────────────────────────────
print("Computing SHAP values for open deals …")
X_open = open_[FULL_COLS].fillna(0)
open_["pred_win_prob"] = cal.predict_proba(X_open)[:, 1]

explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer(X_open).values     # (n_open, n_feats)

# ── 4. GENERATE RECOMMENDATIONS ───────────────────────────────────────────
print("Generating action recommendations …")
recs = []
for i, (idx, row) in enumerate(open_.iterrows()):
    action, desc = recommend_action(row, shap_vals[i], FULL_COLS)
    recs.append({
        "chance_id":     row["chance_id"],
        "business_unit": row["business_unit"],
        "org_country":   row["org_country"],
        "pricelist":     row["pricelist"],
        "amount":        row["amount"],
        "deal_age":      row["deal_age"],
        "pred_win_prob": row["pred_win_prob"],
        "action":        action,
        "action_reason": desc,
        # Top drag feature
        "top_drag_feature": FULL_COLS[np.argmin(shap_vals[i])],
        "top_drag_shap":    float(np.min(shap_vals[i])),
    })

recs_df = pd.DataFrame(recs)
recs_df.to_csv(OUT_DIR / "open_deal_recommendations.csv", index=False)
print(f"  Saved open_deal_recommendations.csv  ({len(recs_df):,} deals)")

# ── 5. ACTION DISTRIBUTION CHART ──────────────────────────────────────────
print("Plotting action distribution …")
action_counts = recs_df["action"].value_counts()

# Stratify by win-prob bucket to show urgency
recs_df["prob_bucket"] = pd.cut(recs_df["pred_win_prob"],
                                 bins=[0, 0.33, 0.65, 1.0],
                                 labels=["Low (<33%)", "Medium (33–65%)", "High (>65%)"])
pivot = (recs_df.groupby(["action", "prob_bucket"], observed=True)
         .size().unstack(fill_value=0))
# Reorder columns
for col in ["Low (<33%)", "Medium (33–65%)", "High (>65%)"]:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[["Low (<33%)", "Medium (33–65%)", "High (>65%)"]].sort_values("Low (<33%)", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
colors = ["#C53030", "#DD6B20", "#276749"]
pivot.plot.barh(stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Number of open deals")
ax.set_title("C4 — Recommended Actions by Win-Probability Tier\n"
             "(red = low-prob deals needing urgent action)", fontsize=10)
ax.legend(title="Win-prob tier")
ax.invert_yaxis()

# Donut: action share
ax = axes[1]
wedge_colors = sns.color_palette("Set2", len(action_counts))
wedges, texts, autotexts = ax.pie(
    action_counts.values, labels=None,
    colors=wedge_colors, autopct="%1.0f%%",
    startangle=90, pctdistance=0.82,
    wedgeprops={"width": 0.5, "edgecolor": "white"},
)
ax.legend(wedges, action_counts.index, loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
ax.set_title("Action Share Across Open Pipeline", fontsize=10)

fig.suptitle("C4 — Next-Best-Action Distribution", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "28_nba_action_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 28_nba_action_distribution.png")

# ── 6. HISTORICAL VALIDATION — WIN RATES BY ACTION PROXY ──────────────────
print("Historical action validation …")

# For each action type, define a proxy observable in closed deals
action_proxies = {
    "Force close or disqualify":     closed["zombie_deal"]             >= 1,
    "Set hard close deadline":       closed["total_days_pushed"]       > 90,
    "Re-engage customer now":        closed["days_since_last_update"]  > 45,
    "Executive sponsor intervention":closed["seq_prob_final_trend"]    < -5
                                      if "seq_prob_final_trend" in closed.columns
                                      else pd.Series(False, index=closed.index),
    "Address objections / stabilise":closed["seq_prob_max_drawdown"]  > 30
                                      if "seq_prob_max_drawdown" in closed.columns
                                      else pd.Series(False, index=closed.index),
    "Advance to next stage":         closed["stage_rank"]              <= 3,
}

val_rows = []
for action_label, mask in action_proxies.items():
    mask = mask.fillna(False)
    n_with    = mask.sum()
    n_without = (~mask).sum()
    wr_with    = closed.loc[mask,  "target_won"].mean() if n_with    > 0 else np.nan
    wr_without = closed.loc[~mask, "target_won"].mean() if n_without > 0 else np.nan
    val_rows.append({
        "Action": action_label,
        "N (condition present)": int(n_with),
        "N (condition absent)":  int(n_without),
        "Win rate (present)":    round(wr_with    * 100, 1) if not np.isnan(wr_with)    else None,
        "Win rate (absent)":     round(wr_without * 100, 1) if not np.isnan(wr_without) else None,
    })

val_df = pd.DataFrame(val_rows)
val_df["Win rate difference (pp)"] = val_df["Win rate (absent)"] - val_df["Win rate (present)"]

fig, ax = plt.subplots(figsize=(11, 5))
val_clean = val_df.dropna(subset=["Win rate difference (pp)"])
bar_colors = ["#C53030" if v > 0 else "#276749"
              for v in val_clean["Win rate difference (pp)"].values]
ax.barh(val_clean["Action"], val_clean["Win rate difference (pp)"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Win rate difference (absent − present) in percentage points")
ax.set_title("C4 — Historical Validation: Win Rate When Risk Signal Present vs Absent\n"
             "Positive bar = deals WITH this risk signal win less often (action is justified)",
             fontsize=10)
for _, row in val_clean.iterrows():
    diff = row["Win rate difference (pp)"]
    ax.annotate(
        f"With: {row['Win rate (present)']}%  |  Without: {row['Win rate (absent)']}%",
        xy=(diff, row["Action"]),
        xytext=(diff + (1 if diff >= 0 else -1), row["Action"]),
        fontsize=7.5, va="center",
        ha="left" if diff >= 0 else "right",
    )

fig.tight_layout()
fig.savefig(OUT_DIR / "29_nba_historical_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 29_nba_historical_validation.png")

# ── 7. PRINT SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("C4 — NEXT-BEST-ACTION SUMMARY")
print("="*65)
print(f"\nOpen deals with recommendations: {len(recs_df):,}")
print("\nAction breakdown:")
for action, count in action_counts.items():
    pct = count / len(recs_df) * 100
    avg_prob = recs_df.loc[recs_df["action"] == action, "pred_win_prob"].mean()
    print(f"  {action:<45}  {count:>5} deals  ({pct:.1f}%)  avg win-prob={avg_prob:.2f}")

print("\nHistorical validation:")
print(val_df[["Action","Win rate (present)","Win rate (absent)",
              "Win rate difference (pp)"]].to_string(index=False))

print("\nDone — next_best_action.py")
