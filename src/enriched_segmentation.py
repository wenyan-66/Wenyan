"""
Enriched Segmentation Analysis
===============================
Combines:
  1. K-Means cluster assignments from clustering_segmentation.py
  2. Win-probability predictions  (forecast_won_lost.parquet, RF model)
  3. Time-to-close predictions    (xgb_aft.json, XGBoost AFT survival model)

For every open opportunity the script produces:
  - forecast_prob_won       (0-1, from Random Forest classifier)
  - pred_days_to_close      (predicted total lifecycle days, from AFT model)
  - pred_days_remaining     (max(0, pred_total - days_already_active))
  - expected_revenue        (amount × forecast_prob_won)
  - pipeline_at_risk        (amount × (1 - forecast_prob_won))

These are then sliced by the three segmentation dimensions:
  business_unit  |  org_country (region)  |  pricelist (product group)

Outputs (in outputs/):
  08_priority_matrix.png              — win-prob vs deal-size scatter, coloured by cluster
  09a_bu_win_prob_heatmap.png         — avg predicted win rate per BU × cluster
  09b_region_win_prob_heatmap.png     — same for region
  09c_pg_win_prob_heatmap.png         — same for product group
  10_urgency_by_segment.png           — avg predicted days remaining per BU / region / PG
  11_pipeline_waterfall.png           — expected revenue vs at-risk value per BU
  12_cluster_forecast_profile.png     — cluster-level avg win-prob & days-remaining
  open_deal_scores.csv                — scored open pipeline (one row per deal)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import xgboost as xgb
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parents[1]
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ── 1. LOAD RAW CHANGELOG & RE-AGGREGATE ──────────────────────────────────
# We re-aggregate to recover pricelist / org_country which the cleaning
# notebook dropped.  The cluster CSV (from clustering_segmentation.py)
# already has these, so we just load that.
print("Loading cluster assignments …")
clusters = pd.read_csv(OUT_DIR / "opportunity_clusters.csv")
print(f"  {len(clusters):,} opportunities with cluster labels")

# ── 2. LOAD WIN-PROBABILITY PREDICTIONS ───────────────────────────────────
print("Loading win-probability predictions …")
wl = pd.read_parquet(ROOT / "forecast_won_lost.parquet")
# keep only the columns we need
wl = wl[["chance_id", "forecast_prob_won"]].copy()
print(f"  {len(wl):,} open deals with forecast_prob_won")

# Merge onto cluster table
df = clusters.merge(wl, on="chance_id", how="left")

# ── 3. COMPUTE AFT FEATURES FOR OPEN DEALS ────────────────────────────────
# The AFT model was trained on creation-time (first snapshot) features.
# We reconstruct these from the raw changelog for open deals.
print("Engineering AFT features from raw changelog …")
raw = pd.read_parquet(ROOT / "chance_changelog.parquet")
raw["registered_dt"] = pd.to_datetime(raw["registered_dt"])
raw["updated_dt"]    = pd.to_datetime(raw["updated_dt"])
raw["saledate"]      = pd.to_datetime(raw["saledate"])

# First snapshot per chance (creation-time state)
first = (
    raw.sort_values(["chance_id", "updated_dt"])
    .groupby("chance_id", sort=False)
    .first()
    .reset_index()
)

# Compute days_until_saledate at registration
first["days_until_saledate_init"] = (
    (first["saledate"] - first["registered_dt"]).dt.total_seconds() / 86_400
).clip(lower=0).fillna(0)

# home_country: 1 if org is in the modal country of the full dataset
top_country = raw["org_country"].value_counts().idxmax()
first["home_country"] = (first["org_country"] == top_country).astype(int)

# Rep-level historical features (computed on all closed deals to avoid leakage
# at scoring time — we only score open deals, so this is fine)
closed = raw[raw["status"].isin(["Won", "Lost"])].copy()
closed_agg = (
    closed.sort_values(["chance_id", "updated_dt"])
    .groupby("chance_id")
    .last()
    .reset_index()[["chance_id", "responsible", "org_id", "status", "amount"]]
)
closed_agg["is_won"] = (closed_agg["status"] == "Won").astype(int)

# rep_closing_rate: win rate per rep on closed deals
rep_stats = (
    closed_agg.groupby("responsible")
    .agg(rep_closing_rate=("is_won", "mean"),
         rep_amount=("amount", "sum"))
    .reset_index()
)

# customer_win_ratio & is_repeat_customer per org
cust_stats = (
    closed_agg.groupby("org_id")
    .agg(customer_win_ratio=("is_won", "mean"),
         n_deals=("chance_id", "count"))
    .reset_index()
)
cust_stats["is_repeat_customer"] = (cust_stats["n_deals"] > 1).astype(int)

# Merge rep / customer stats onto first snapshots
first = first.merge(rep_stats, on="responsible", how="left")
first = first.merge(cust_stats[["org_id", "customer_win_ratio", "is_repeat_customer"]],
                    on="org_id", how="left")

# deal_share_rep: this deal's amount / rep's total amount
first["deal_share_rep"] = (
    first["amount"] / first["rep_amount"].replace(0, np.nan)
).fillna(0).clip(0, 1)

# Fill remaining NaNs with defaults
first["rep_closing_rate"]   = first["rep_closing_rate"].fillna(0.5)
first["rep_amount"]         = first["rep_amount"].fillna(first["amount"])
first["customer_win_ratio"] = first["customer_win_ratio"].fillna(0.5)
first["is_repeat_customer"] = first["is_repeat_customer"].fillna(0)

# Rename columns to match AFT model expectations (_init suffix for creation-time)
first["probability_init"]              = first["probability"]
first["log_amount_init"]               = np.log1p(first["amount"].clip(lower=0).fillna(0))
first["stage_en_init"]                 = first["stage_en"]
first["has_products_and_services_init"] = first["has_products_and_services"]

# ── 4. SCORE OPEN DEALS WITH AFT MODEL ────────────────────────────────────
print("Scoring open deals with AFT model …")

AFT_FEATURES = [
    "probability_init", "log_amount_init", "stage_en_init",
    "days_until_saledate_init", "has_products_and_services_init",
    "business_unit", "org_category_en", "chance_type_en",
    "home_country", "intercompany_flag",
    "rep_closing_rate", "rep_amount", "deal_share_rep",
    "customer_win_ratio", "is_repeat_customer",
]
CAT_COLS = ["stage_en_init", "business_unit", "org_category_en", "chance_type_en"]

# Load model artifacts
aft_model   = xgb.Booster()
aft_model.load_model(ROOT / "xgb_aft.json")
imputer     = joblib.load(ROOT / "imputer.joblib")
enc         = joblib.load(ROOT / "ordinal_encoder_aft.joblib")

# Restrict to open deals that have all first-snapshot features
open_ids  = df.loc[df["final_status"] == "Open", "chance_id"].values
open_first = first[first["chance_id"].isin(open_ids)].copy()

# Check available features
missing_feats = [f for f in AFT_FEATURES if f not in open_first.columns]
if missing_feats:
    print(f"  WARNING: missing AFT features: {missing_feats}")
    for f in missing_feats:
        open_first[f] = 0

X_aft = open_first[["chance_id"] + AFT_FEATURES].copy()

# Encode categoricals using the saved ordinal encoder
for i, col in enumerate(CAT_COLS):
    known_cats = list(enc.categories_[i])
    X_aft[col] = X_aft[col].where(X_aft[col].isin(known_cats), other=known_cats[0])

X_aft[CAT_COLS] = enc.transform(X_aft[CAT_COLS])

# Impute remaining numeric columns with median (the saved imputer was fitted
# on the raw/wider feature set with different column names, so we impute directly)
num_cols = [f for f in AFT_FEATURES if f not in CAT_COLS]
for col in num_cols:
    median_val = X_aft[col].median()
    X_aft[col] = X_aft[col].fillna(median_val if not np.isnan(median_val) else 0)

# Predict with AFT model
dmat = xgb.DMatrix(X_aft[AFT_FEATURES].values, feature_names=AFT_FEATURES)
pred_total_days = aft_model.predict(dmat)

aft_preds = pd.DataFrame({
    "chance_id":        X_aft["chance_id"].values,
    "pred_total_days":  pred_total_days,
})

print(f"  Scored {len(aft_preds):,} open deals with AFT model")
print(f"  Predicted total days — median: {np.median(pred_total_days):.0f}  "
      f"p25: {np.percentile(pred_total_days, 25):.0f}  "
      f"p75: {np.percentile(pred_total_days, 75):.0f}")

# ── 5. MERGE ALL PREDICTIONS INTO MAIN TABLE ──────────────────────────────
print("Merging predictions …")

df = df.merge(aft_preds, on="chance_id", how="left")

# Predicted days remaining (capped at 0)
df["pred_days_remaining"] = (
    (df["pred_total_days"] - df["days_active"]).clip(lower=0)
)

# Financial signals (open deals only)
df["expected_revenue"]  = df["final_amount"] * df["forecast_prob_won"]
df["pipeline_at_risk"]  = df["final_amount"] * (1 - df["forecast_prob_won"])

# Restrict open-deal view
open_df = df[df["final_status"] == "Open"].copy()
print(f"  Open deals with win-prob: {open_df['forecast_prob_won'].notna().sum():,}")
print(f"  Open deals with AFT pred: {open_df['pred_total_days'].notna().sum():,}")

# ── 6. CLUSTER LABEL MAP ───────────────────────────────────────────────────
# Derive labels from profile (same logic as clustering_segmentation.py)
profile = (
    df.groupby("cluster")[["final_amount", "final_probability", "days_active"]]
    .mean()
)
# Simple labelling based on original profile output
cluster_label_map = {int(c): lbl for c, lbl in
    df[["cluster", "cluster_label"]].drop_duplicates()
    .set_index("cluster")["cluster_label"].items()
}

# ── 7. CHART HELPERS ───────────────────────────────────────────────────────
PALETTE = sns.color_palette("Set1", 5)

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── 8. PRIORITY MATRIX ────────────────────────────────────────────────────
print("\n── Priority matrix (win-prob × deal-size) ──")
plot_df = open_df.dropna(subset=["forecast_prob_won", "final_amount"]).copy()
plot_df["log_amount"] = np.log1p(plot_df["final_amount"])

fig, ax = plt.subplots(figsize=(11, 7))
for c, color in zip(sorted(plot_df["cluster"].unique()), PALETTE):
    sub = plot_df[plot_df["cluster"] == c]
    sc = ax.scatter(
        sub["forecast_prob_won"], sub["log_amount"],
        s=np.clip(sub.get("pred_days_remaining", pd.Series(30)).fillna(30), 5, 500) * 0.5 + 10,
        alpha=0.45, color=color,
        label=cluster_label_map.get(c, f"C{c}"),
    )

ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% win-prob")
ax.axvline(0.65, color="red", linestyle=":", linewidth=0.8, label="65% threshold")
ax.set_xlabel("Predicted Win Probability", fontsize=11)
ax.set_ylabel("Log Deal Amount", fontsize=11)
ax.set_title("Open Deal Priority Matrix\n"
             "(size = predicted days remaining; colour = cluster)", fontsize=12)
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
_save(fig, "08_priority_matrix.png")


# ── 9. WIN-PROB HEATMAPS BY SEGMENT × CLUSTER ─────────────────────────────
def win_prob_heatmap(src, dim_col, dim_label, top_n, filename):
    top_vals = src[dim_col].value_counts().head(top_n).index
    sub = src[src[dim_col].isin(top_vals)].dropna(subset=["forecast_prob_won"])
    if sub.empty:
        print(f"  SKIP {filename}: no data")
        return
    pivot = (
        sub.groupby([dim_col, "cluster"])["forecast_prob_won"]
        .mean().unstack() * 100
    )
    pivot.columns = [cluster_label_map.get(int(c), f"C{c}") for c in pivot.columns]
    pivot.index.name = dim_label

    fig, ax = plt.subplots(figsize=(max(10, 5 * len(pivot.columns) // 2),
                                    max(5, top_n * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, vmin=0, vmax=100,
                cbar_kws={"label": "Avg predicted win prob (%)"})
    ax.set_title(f"Predicted Win Probability (%) by {dim_label} × Cluster\n"
                 f"(open deals only)", fontsize=12)
    _save(fig, filename)

print("\n── Win-prob heatmaps ──")
win_prob_heatmap(open_df, "business_unit", "Business Unit",
                 top_n=7,  filename="09a_bu_win_prob_heatmap.png")
win_prob_heatmap(open_df, "org_country",   "Region",
                 top_n=20, filename="09b_region_win_prob_heatmap.png")
win_prob_heatmap(open_df, "pricelist",     "Product Group",
                 top_n=15, filename="09c_pg_win_prob_heatmap.png")


# ── 10. URGENCY BY SEGMENT ────────────────────────────────────────────────
print("\n── Urgency charts ──")

def urgency_bar(src, dim_col, dim_label, top_n, filename):
    top_vals = src[dim_col].value_counts().head(top_n).index
    sub = src[src[dim_col].isin(top_vals)].dropna(subset=["pred_days_remaining"])
    if sub.empty:
        print(f"  SKIP {filename}: no data")
        return

    stats = (
        sub.groupby(dim_col)["pred_days_remaining"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "Avg days", "median": "Median days", "count": "N"})
        .sort_values("Avg days")          # shortest first = most urgent
    )

    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.55), 5))
    bars = ax.bar(range(len(stats)), stats["Avg days"],
                  color=sns.color_palette("YlOrRd_r", len(stats)))
    ax.plot(range(len(stats)), stats["Median days"], "D--",
            color="#1a1a1a", markersize=5, label="Median")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Predicted days remaining")
    ax.set_title(f"Deal Urgency by {dim_label}\n"
                 f"(shorter = closer to closing; red = most urgent)", fontsize=12)
    ax.legend()
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(i, row["Avg days"] + 1, f"n={int(row['N'])}", ha="center",
                fontsize=7, color="#444")
    _save(fig, filename)

urgency_bar(open_df, "business_unit", "Business Unit",
            top_n=7,  filename="10a_urgency_bu.png")
urgency_bar(open_df, "org_country",   "Region",
            top_n=20, filename="10b_urgency_region.png")
urgency_bar(open_df, "pricelist",     "Product Group",
            top_n=15, filename="10c_urgency_pg.png")


# ── 11. PIPELINE WATERFALL BY BUSINESS UNIT ───────────────────────────────
print("\n── Pipeline waterfall ──")
waterfall_df = (
    open_df.dropna(subset=["forecast_prob_won", "final_amount"])
    .groupby("business_unit")
    .agg(
        expected_revenue  = ("expected_revenue", "sum"),
        pipeline_at_risk  = ("pipeline_at_risk", "sum"),
        n_deals           = ("chance_id", "count"),
    )
    .sort_values("expected_revenue", ascending=False)
)
waterfall_df["total_pipeline"] = (
    waterfall_df["expected_revenue"] + waterfall_df["pipeline_at_risk"]
)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(waterfall_df))
ax.bar(x, waterfall_df["expected_revenue"] / 1e6,
       label="Expected Revenue (won)", color="#276749")
ax.bar(x, waterfall_df["pipeline_at_risk"] / 1e6,
       bottom=waterfall_df["expected_revenue"] / 1e6,
       label="Pipeline at Risk (loss)", color="#C53030", alpha=0.75)
ax.set_xticks(list(x))
ax.set_xticklabels(waterfall_df.index, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Amount (M)")
ax.set_title("Open Pipeline: Expected Revenue vs At-Risk Amount by Business Unit",
             fontsize=12)
ax.legend()
for i, (idx, row) in enumerate(waterfall_df.iterrows()):
    ax.text(i, row["total_pipeline"] / 1e6 + 0.05,
            f"n={int(row['n_deals'])}", ha="center", fontsize=7)
_save(fig, "11_pipeline_waterfall.png")


# ── 12. ENRICHED CLUSTER PROFILE ─────────────────────────────────────────
print("\n── Cluster forecast profile ──")
clust_profile = (
    open_df.groupby("cluster_label")
    .agg(
        n_open             = ("chance_id",        "count"),
        avg_win_prob       = ("forecast_prob_won", "mean"),
        avg_days_remaining = ("pred_days_remaining","mean"),
        total_expected_rev = ("expected_revenue",  "sum"),
        total_at_risk      = ("pipeline_at_risk",  "sum"),
    )
    .round({"avg_win_prob": 3, "avg_days_remaining": 1})
)
clust_profile["total_expected_rev_M"] = (clust_profile["total_expected_rev"] / 1e6).round(2)
clust_profile["total_at_risk_M"]      = (clust_profile["total_at_risk"]      / 1e6).round(2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: win prob by cluster
cp = clust_profile.sort_values("avg_win_prob", ascending=False)
axes[0].barh(cp.index, cp["avg_win_prob"] * 100,
             color=sns.color_palette("RdYlGn", len(cp)))
axes[0].set_xlabel("Avg predicted win probability (%)")
axes[0].set_title("Win Probability per Cluster\n(open deals)")
axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# Right: urgency (days remaining) by cluster
cp2 = clust_profile.sort_values("avg_days_remaining")
axes[1].barh(cp2.index, cp2["avg_days_remaining"],
             color=sns.color_palette("YlOrRd_r", len(cp2)))
axes[1].set_xlabel("Avg predicted days remaining")
axes[1].set_title("Urgency per Cluster\n(shorter = act sooner)")

_save(fig, "12_cluster_forecast_profile.png")


# ── 13. SUMMARY TABLES ────────────────────────────────────────────────────
print("\nWriting CSV summaries …")

# Open deal scores
export_cols = [
    "chance_id", "business_unit", "org_country", "pricelist",
    "final_status", "final_amount", "days_active",
    "cluster", "cluster_label",
    "forecast_prob_won", "pred_total_days", "pred_days_remaining",
    "expected_revenue", "pipeline_at_risk",
]
available = [c for c in export_cols if c in open_df.columns]
open_df[available].to_csv(OUT_DIR / "open_deal_scores.csv", index=False)
print("  Saved open_deal_scores.csv")

# Cluster-level forecast summary
clust_profile.to_csv(OUT_DIR / "cluster_forecast_summary.csv")
print("  Saved cluster_forecast_summary.csv")

# BU pipeline summary
waterfall_df.drop(columns=["total_pipeline"]).to_csv(
    OUT_DIR / "bu_pipeline_summary.csv")
print("  Saved bu_pipeline_summary.csv")


# ── 14. PRINT SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ENRICHED CLUSTER FORECAST PROFILE (open deals)")
print("=" * 70)
pd.set_option("display.float_format", "{:.3f}".format)
print(clust_profile.to_string())

print("\n" + "=" * 70)
print("BUSINESS UNIT PIPELINE SUMMARY (M = millions)")
print("=" * 70)
print(waterfall_df[["n_deals","expected_revenue","pipeline_at_risk"]].to_string())

print(f"\n✓ All outputs written to: {OUT_DIR}")
print("Done.")
