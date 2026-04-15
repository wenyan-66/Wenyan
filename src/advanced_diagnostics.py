"""
B1  Trajectory Clustering         — K-Means on resampled probability curves
B2  Uplift Modeling (T-Learner)   — treatment proxy = saledate was pushed
B5  Rep Performance Analysis      — risk-adjusted win rate vs model expectation
B6  Concept Drift Monitoring      — PSI on feature distributions over time
C2  Anomaly Detection             — Isolation Forest flags unusual deals
C3  Customer Lifetime Value       — expected future revenue per account

Outputs:
  outputs/19_trajectory_clusters.png
  outputs/20_anomaly_scores.png
  outputs/21_rep_performance.png
  outputs/22_drift_psi.png
  outputs/23_uplift_distribution.png
  outputs/24_clv_distribution.png
  outputs/anomaly_deals.csv
  outputs/rep_performance.csv
  outputs/clv_scores.csv
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
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from feature_engineering import (load_raw, engineer_features,
                                  NUMERIC_FEATURES, CAT_FEATURES, ALL_FEATURES)

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
SEED    = 42


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── LOAD ───────────────────────────────────────────────────────────────────
print("Loading data …")
raw = load_raw()
df  = engineer_features(raw)
for col in CAT_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))


# ══════════════════════════════════════════════════════════════════════════
# B1  TRAJECTORY CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
print("\n── B1: Trajectory Clustering ──")

# Extract probability sequences per deal (need raw changelog)
raw_sorted = raw.sort_values(["chance_id", "updated_dt"])
traj_rows  = []

for cid, grp in raw_sorted.groupby("chance_id", sort=False):
    if len(grp) < 4:
        continue
    probs = grp["probability"].values.astype(float)
    # Normalise time to [0, 1]
    t_norm = np.linspace(0, 1, len(probs))
    # Resample to 10 equally-spaced points via linear interp
    t_grid = np.linspace(0, 1, 10)
    p_resampled = np.interp(t_grid, t_norm, probs)
    status = grp["status"].iloc[-1]
    traj_rows.append({
        "chance_id": cid,
        "status": status,
        **{f"t{i}": p_resampled[i] for i in range(10)},
    })

traj = pd.DataFrame(traj_rows)
print(f"  Trajectories (≥4 updates): {len(traj):,}")

T_COLS = [f"t{i}" for i in range(10)]
X_traj = traj[T_COLS].fillna(0).values
scaler_t = StandardScaler()
X_traj_s = scaler_t.fit_transform(X_traj)

K_TRAJ = 6
km_traj = KMeans(n_clusters=K_TRAJ, random_state=SEED, n_init=15)
traj["traj_cluster"] = km_traj.fit_predict(X_traj_s)

# Plot typical shapes
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
axes = axes.flatten()
t_axis = np.linspace(0, 100, 10)
status_colors = {"Won": "#276749", "Lost": "#C53030", "Open": "#2B6CB0", "Interrupted": "#744210"}

for c in range(K_TRAJ):
    ax = axes[c]
    sub = traj[traj["traj_cluster"] == c]
    # Draw sample of individual trajectories (faint)
    sample = sub.sample(min(50, len(sub)), random_state=SEED)
    for _, row in sample.iterrows():
        col = status_colors.get(row["status"], "gray")
        ax.plot(t_axis, row[T_COLS].values, color=col, alpha=0.12, linewidth=0.8)
    # Mean trajectory per outcome
    for outcome in ["Won", "Lost", "Open"]:
        sub_o = sub[sub["status"] == outcome]
        if len(sub_o) > 0:
            ax.plot(t_axis, sub_o[T_COLS].mean().values,
                    color=status_colors[outcome], linewidth=2.2,
                    label=f"{outcome} (n={len(sub_o):,})")
    win_rate = (sub["status"] == "Won").mean() * 100
    ax.set_title(f"Cluster {c}  |  n={len(sub):,}  |  WR={win_rate:.0f}%", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Deal lifecycle (%)" if c >= 3 else "")
    ax.set_ylabel("Probability" if c % 3 == 0 else "")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

fig.suptitle("B1 — Probability Trajectory Clusters\n"
             "Each panel = a distinct deal lifecycle pattern", fontsize=12)
_save(fig, "19_trajectory_clusters.png")

# ══════════════════════════════════════════════════════════════════════════
# C2  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n── C2: Anomaly Detection ──")

feats = [f for f in NUMERIC_FEATURES if f in df.columns]
X_all = df[feats].fillna(0)

iso = IsolationForest(n_estimators=200, contamination=0.05,
                      random_state=SEED, n_jobs=-1)
df["anomaly_score"] = iso.fit_predict(X_all)          # -1 = anomaly
df["anomaly_raw"]   = iso.score_samples(X_all)        # lower = more anomalous

anomalies = df[df["anomaly_score"] == -1].copy()
print(f"  Flagged {len(anomalies):,} anomalous deals ({len(anomalies)/len(df)*100:.1f}%)")
print(f"  Of anomalies: {(anomalies['status']=='Won').mean()*100:.1f}% Won | "
      f"{(anomalies['status']=='Lost').mean()*100:.1f}% Lost | "
      f"{(anomalies['status']=='Open').mean()*100:.1f}% Open")

# What makes them anomalous?
feat_means_normal  = df[df["anomaly_score"] ==  1][feats].mean()
feat_means_anomaly = anomalies[feats].mean()
anomaly_diff = ((feat_means_anomaly - feat_means_normal) /
                feat_means_normal.replace(0, 1)).sort_values(key=abs, ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["anomaly_raw"],   bins=60, color="#2B6CB0", alpha=0.6, label="Normal")
axes[0].hist(anomalies["anomaly_raw"], bins=30, color="#C53030", alpha=0.7, label="Anomaly")
axes[0].set_xlabel("Anomaly score (lower = more anomalous)")
axes[0].set_ylabel("Count")
axes[0].set_title("C2 — Anomaly Score Distribution")
axes[0].legend()

top_diff = anomaly_diff.head(12)
colors = ["#C53030" if v > 0 else "#276749" for v in top_diff.values]
axes[1].barh(top_diff.index, top_diff.values, color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("Top Features Driving Anomaly\n(red = anomalies have higher value)")
axes[1].set_xlabel("Relative difference vs normal deals")

_save(fig, "20_anomaly_scores.png")

# Export anomaly list
anom_export = anomalies[["chance_id","business_unit","org_country","pricelist",
                          "status","amount","deal_age","anomaly_raw"]].copy()
anom_export = anom_export.sort_values("anomaly_raw")
anom_export.to_csv(OUT_DIR / "anomaly_deals.csv", index=False)
print("  Saved anomaly_deals.csv")


# ══════════════════════════════════════════════════════════════════════════
# B5  REP PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n── B5: Rep Performance Analysis ──")

# Train XGB on closed deals
closed = df[df["is_closed"] == 1].copy()
feats_m = [f for f in ALL_FEATURES if f in df.columns]
X_cl = closed[feats_m].fillna(0)
y_cl = closed["target_won"]

xgb_rep = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                         eval_metric="logloss", random_state=SEED,
                         verbosity=0, n_jobs=-1)
xgb_rep.fit(X_cl, y_cl)
closed["pred_win_prob"] = xgb_rep.predict_proba(X_cl)[:, 1]

# Rep-level comparison: actual win rate vs model-expected
rep_perf = (closed.groupby("responsible")
            .agg(
                n_deals         = ("chance_id",      "count"),
                actual_wr       = ("target_won",     "mean"),
                expected_wr     = ("pred_win_prob",  "mean"),
                total_amount_M  = ("amount",         lambda x: x.sum()/1e6),
            )
            .reset_index())
rep_perf["outperformance"] = rep_perf["actual_wr"] - rep_perf["expected_wr"]

# Keep reps with ≥15 closed deals for statistical reliability
rep_perf = rep_perf[rep_perf["n_deals"] >= 15].sort_values("outperformance", ascending=False)
print(f"  Reps with ≥15 closed deals: {len(rep_perf):,}")
print(f"  Top outperformer alpha: {rep_perf['outperformance'].max():+.3f}")
print(f"  Worst underperformer alpha: {rep_perf['outperformance'].min():+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: actual vs expected
ax = axes[0]
sc = ax.scatter(rep_perf["expected_wr"]*100, rep_perf["actual_wr"]*100,
                c=rep_perf["outperformance"], cmap="RdYlGn",
                s=rep_perf["n_deals"]*3, alpha=0.7, vmin=-0.3, vmax=0.3)
ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Perfect calibration")
plt.colorbar(sc, ax=ax, label="Outperformance (actual − expected)")
ax.set_xlabel("Model-expected win rate (%)")
ax.set_ylabel("Actual win rate (%)")
ax.set_title("B5 — Rep Performance vs Model Expectation\n"
             "(size = deal count; green = outperformer; red = underperformer)", fontsize=10)
ax.legend()

# Bar: top/bottom 15 by outperformance
ax = axes[1]
top15 = pd.concat([rep_perf.head(8), rep_perf.tail(7)]).reset_index(drop=True)
colors = ["#276749" if v > 0 else "#C53030" for v in top15["outperformance"]]
ax.barh(range(len(top15)), top15["outperformance"]*100, color=colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([r[:14] for r in top15["responsible"]], fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Win-rate outperformance (pp)")
ax.set_title("Top & Bottom Reps by Risk-Adjusted Performance")

_save(fig, "21_rep_performance.png")

rep_perf.to_csv(OUT_DIR / "rep_performance.csv", index=False)
print("  Saved rep_performance.csv")


# ══════════════════════════════════════════════════════════════════════════
# B6  CONCEPT DRIFT MONITORING  (PSI)
# ══════════════════════════════════════════════════════════════════════════
print("\n── B6: Concept Drift Monitoring (PSI) ──")

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-6
    bns   = np.linspace(min(expected.min(), actual.min()),
                        max(expected.max(), actual.max()) + eps, bins + 1)
    exp_p = np.histogram(expected, bins=bns)[0] / (len(expected) + eps)
    act_p = np.histogram(actual,   bins=bns)[0] / (len(actual)   + eps)
    exp_p = np.where(exp_p == 0, eps, exp_p)
    act_p = np.where(act_p == 0, eps, act_p)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

# Split: "old" baseline = registered before 2023, "recent" = 2024+
old    = df[df["year_registered"] <  2023]
recent = df[df["year_registered"] >= 2024]
print(f"  Old baseline: {len(old):,} deals  |  Recent: {len(recent):,} deals")

psi_scores = {}
numeric_feats = [f for f in NUMERIC_FEATURES if f in df.columns]
for feat in numeric_feats:
    if old[feat].std() > 0:
        psi_scores[feat] = psi(old[feat].fillna(0).values,
                               recent[feat].fillna(0).values)

psi_df = (pd.Series(psi_scores, name="PSI")
          .sort_values(ascending=False)
          .reset_index().rename(columns={"index": "feature"}))
psi_df["status"] = psi_df["PSI"].apply(
    lambda x: "High drift" if x > 0.25 else ("Moderate" if x > 0.1 else "Stable"))

print(f"  High drift features  (PSI>0.25): {(psi_df['PSI']>0.25).sum()}")
print(f"  Moderate drift (PSI 0.1-0.25)  : {((psi_df['PSI']>0.1)&(psi_df['PSI']<=0.25)).sum()}")
print(f"  Stable         (PSI<0.10)       : {(psi_df['PSI']<0.1).sum()}")

fig, ax = plt.subplots(figsize=(10, 7))
status_pal = {"High drift": "#C53030", "Moderate": "#DD6B20", "Stable": "#276749"}
bar_colors = [status_pal[s] for s in psi_df["status"]]
ax.barh(psi_df["feature"], psi_df["PSI"], color=bar_colors)
ax.axvline(0.1,  color="orange", linestyle="--", linewidth=1, label="Moderate (0.10)")
ax.axvline(0.25, color="red",    linestyle="--", linewidth=1, label="High drift (0.25)")
ax.set_xlabel("Population Stability Index (PSI)")
ax.set_title("B6 — Feature Drift: Old Deals (pre-2023) vs Recent (2024+)\n"
             "Red bars require model retraining attention", fontsize=11)
ax.legend()
_save(fig, "22_drift_psi.png")


# ══════════════════════════════════════════════════════════════════════════
# B2  UPLIFT MODELING — T-LEARNER
# ══════════════════════════════════════════════════════════════════════════
print("\n── B2: Uplift Modeling (T-Learner) ──")

# Treatment proxy: saledate was actively pushed (intervention)
closed["treatment"] = (closed["total_days_pushed"] > 0).astype(int)
t_rate = closed["treatment"].mean()
print(f"  Treatment rate (saledate pushed): {t_rate:.2%}")

feats_u = [f for f in NUMERIC_FEATURES if f in closed.columns]
X_t0 = closed.loc[closed["treatment"] == 0, feats_u].fillna(0)
y_t0 = closed.loc[closed["treatment"] == 0, "target_won"]
X_t1 = closed.loc[closed["treatment"] == 1, feats_u].fillna(0)
y_t1 = closed.loc[closed["treatment"] == 1, "target_won"]

print(f"  Control (T=0): {len(X_t0):,}  |  Treated (T=1): {len(X_t1):,}")

m0 = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4,
                    eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1)
m1 = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4,
                    eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1)
m0.fit(X_t0, y_t0)
m1.fit(X_t1, y_t1)

# ITE on all closed deals: P(won|X,T=1) - P(won|X,T=0)
X_all_cl = closed[feats_u].fillna(0)
closed["p_t0"]  = m0.predict_proba(X_all_cl)[:, 1]
closed["p_t1"]  = m1.predict_proba(X_all_cl)[:, 1]
closed["ite"]   = closed["p_t1"] - closed["p_t0"]

print(f"  Mean ITE (avg uplift from intervention): {closed['ite'].mean():+.4f}")
print(f"  Deals where intervention helps (ITE>0) : {(closed['ite']>0).mean():.1%}")
print(f"  Deals where intervention hurts (ITE<0) : {(closed['ite']<0).mean():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(closed.loc[closed["treatment"]==0,"ite"], bins=50,
        color="#2B6CB0", alpha=0.6, label="Control (no push)")
ax.hist(closed.loc[closed["treatment"]==1,"ite"], bins=50,
        color="#276749", alpha=0.6, label="Treated (saledate pushed)")
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Individual Treatment Effect (ITE)")
ax.set_ylabel("Count")
ax.set_title("B2 — Uplift Distribution by Treatment Group\n"
             "ITE > 0 = intervention likely helps", fontsize=10)
ax.legend()

# ITE by stage rank (does stage matter for whether intervention helps?)
ax = axes[1]
ite_by_stage = closed.groupby("stage_rank")["ite"].mean()
colors = ["#276749" if v > 0 else "#C53030" for v in ite_by_stage.values]
ax.bar(ite_by_stage.index, ite_by_stage.values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Stage Rank (1=Lead → 9=Order received)")
ax.set_ylabel("Mean ITE")
ax.set_title("Average Uplift by Stage\n"
             "Green = intervention helps at that stage")

_save(fig, "23_uplift_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# C3  CUSTOMER LIFETIME VALUE
# ══════════════════════════════════════════════════════════════════════════
print("\n── C3: Customer Lifetime Value ──")

# Per-account metrics from closed deals
closed_full = df[df["is_closed"] == 1].copy()
clv = (closed_full.groupby("org_id")
       .agg(
           n_deals        = ("chance_id",    "count"),
           n_won          = ("target_won",   "sum"),
           win_rate       = ("target_won",   "mean"),
           avg_deal_value = ("amount",       "mean"),
           total_revenue  = ("amount",       lambda x:
                             (x * (closed_full.loc[x.index,"target_won"])).sum()),
           org_country    = ("org_country",  "last"),
           pricelist      = ("pricelist",    "last"),
       )
       .reset_index())

# CLV estimate: expected deals/year × avg value × win rate × 2-year horizon
# Expected deals/year = historical deal rate
# We know deals are from ~2019-2025 ≈ 6 years
HORIZON_YEARS = 2
OBS_YEARS     = 6
clv["deal_rate_yr"]  = clv["n_deals"] / OBS_YEARS
clv["clv_estimate"]  = (clv["deal_rate_yr"] * clv["avg_deal_value"]
                        * clv["win_rate"] * HORIZON_YEARS)
clv = clv.sort_values("clv_estimate", ascending=False).reset_index(drop=True)

print(f"  Unique accounts analysed: {len(clv):,}")
print(f"  Top-10 accounts CLV: {clv['clv_estimate'].head(10).sum()/1e6:.1f}M")
print(f"  Top-100 accounts CLV: {clv['clv_estimate'].head(100).sum()/1e6:.1f}M")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# CLV distribution (log scale)
ax = axes[0]
clv_pos = clv[clv["clv_estimate"] > 0]["clv_estimate"]
ax.hist(np.log10(clv_pos + 1), bins=40, color="#2B6CB0", edgecolor="white")
ax.set_xlabel("log10(CLV estimate + 1)")
ax.set_ylabel("Number of accounts")
ax.set_title("C3 — Customer Lifetime Value Distribution\n"
             "(2-year horizon, based on historical deal rate × win rate)", fontsize=10)

# Lorenz curve (inequality of value)
ax = axes[1]
clv_sorted = np.sort(clv["clv_estimate"].values)
cum_clv = np.cumsum(clv_sorted) / (clv_sorted.sum() + 1e-9)
cum_acct = np.linspace(0, 1, len(clv_sorted))
ax.plot(cum_acct*100, cum_clv*100, color="#2B6CB0", linewidth=2)
ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Perfect equality")
# Find: top X% of accounts drive Y% of CLV
top10_idx = int(len(clv_sorted) * 0.9)
top10_share = (1 - cum_clv[top10_idx]) * 100
ax.annotate(f"Top 10% accounts\n= {top10_share:.0f}% of CLV",
            xy=(90, cum_clv[top10_idx]*100),
            xytext=(60, 30), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.set_xlabel("Cumulative % of accounts")
ax.set_ylabel("Cumulative % of CLV")
ax.set_title("Lorenz Curve: Account CLV Concentration")
ax.legend()
ax.grid(alpha=0.3)

_save(fig, "24_clv_distribution.png")

clv.head(200).to_csv(OUT_DIR / "clv_scores.csv", index=False)
print("  Saved clv_scores.csv (top 200 accounts)")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ADVANCED DIAGNOSTICS — SUMMARY")
print("="*65)

# Trajectory cluster composition
print("\nB1 — Trajectory cluster outcomes:")
tc = traj.groupby("traj_cluster")["status"].value_counts(normalize=True).unstack().fillna(0)
tc["n"] = traj.groupby("traj_cluster").size()
print(tc.round(3).to_string())

print(f"\nC2 — Anomaly rate: {len(anomalies)/len(df)*100:.1f}% of all deals")
print(f"B5 — Rep outperformance range: "
      f"{rep_perf['outperformance'].min():+.2%} to "
      f"{rep_perf['outperformance'].max():+.2%}")
print(f"B6 — Features with high drift: {(psi_df['PSI']>0.25).sum()}")
print(f"B2 — Avg uplift from intervention: {closed['ite'].mean():+.4f}")
print(f"C3 — Top 10% of accounts = {top10_share:.0f}% of estimated future CLV")

print("\nDone — advanced_diagnostics.py")
