"""
RQ4 — Model-Driven Segmentation by Business Unit, Region, Product Group
=========================================================================
Research Question: How can the model support segmentation by dimensions
such as business unit, region, and product group?

Answer approach:
  The win-probability model (RQ1) and time-to-close model (RQ2) each assign
  a continuous score to every open deal.  Aggregating these scores across the
  three segmentation dimensions reveals:

    RQ1 signal → WHICH segments are most likely to win (probability profile)
    RQ2 signal → HOW SOON each segment is expected to close (urgency profile)
    RQ1 × RQ2  → Strategic quadrant: where to act now vs later vs exit

Data sources:
  opportunity_clusters.csv   — cluster labels + original dimension labels
  calibrated_open_scores.csv — pred_win_prob_cal  (RQ1, calibrated XGBoost)
  open_deal_scores.csv       — pred_days_remaining (RQ2, XGBoost AFT)

Coverage: 4,845 open deals with both RQ1 and RQ2 scores

Outputs:
  outputs/30_rq4_bu_model_segments.png
  outputs/31_rq4_region_model_segments.png
  outputs/32_rq4_pg_model_segments.png
  outputs/33_rq4_strategic_quadrant.png
  outputs/34_rq4_win_tier_composition.png
  outputs/rq4_segment_model_summary.csv
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
from pathlib import Path

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"


# ── 1. LOAD & MERGE ────────────────────────────────────────────────────────
print("Loading model outputs …")

# Base: original dimension labels + cluster assignments (all open deals)
clusters = pd.read_csv(OUT_DIR / "opportunity_clusters.csv")
open_cl  = clusters[clusters["final_status"] == "Open"].copy()

# RQ1: calibrated win-probability — pred_win_prob_cal (4,845 deals)
cal = pd.read_csv(OUT_DIR / "calibrated_open_scores.csv")[
    ["chance_id", "pred_win_prob_cal"]
]

# RQ2: AFT predicted days remaining (4,845 deals)
scored = pd.read_csv(OUT_DIR / "open_deal_scores.csv")[
    ["chance_id", "pred_days_remaining", "pred_total_days"]
]

df = (open_cl
      .merge(cal,    on="chance_id", how="left")
      .merge(scored, on="chance_id", how="left"))

df = df.rename(columns={"pred_win_prob_cal": "win_prob",
                         "final_amount":      "amount"})

# Derived financial signals
df["expected_revenue"] = df["amount"] * df["win_prob"]
df["pipeline_at_risk"] = df["amount"] * (1 - df["win_prob"])

# Win-probability tier (RQ1 classification)
df["win_tier"] = pd.cut(
    df["win_prob"],
    bins=[0, 0.33, 0.65, 1.0],
    labels=["Low (<33%)", "Medium (33–65%)", "High (>65%)"],
    include_lowest=True,
)

print(f"  Open deals loaded  : {len(df):,}")
print(f"  With RQ1 win_prob  : {df['win_prob'].notna().sum():,}")
print(f"  With RQ2 days_rem  : {df['pred_days_remaining'].notna().sum():,}")


# ── 2. SEGMENT PROFILE BUILDER ─────────────────────────────────────────────
def build_profile(src, dim_col, top_n):
    top_vals = src[dim_col].value_counts().head(top_n).index
    sub = src[src[dim_col].isin(top_vals)]
    profile = (
        sub.groupby(dim_col)
        .agg(
            n_deals            = ("chance_id",           "count"),
            avg_win_prob       = ("win_prob",             "mean"),
            std_win_prob       = ("win_prob",             "std"),
            avg_days_remaining = ("pred_days_remaining",  "mean"),
            med_days_remaining = ("pred_days_remaining",  "median"),
            total_amount       = ("amount",               "sum"),
            expected_revenue   = ("expected_revenue",     "sum"),
            pipeline_at_risk   = ("pipeline_at_risk",     "sum"),
        )
        .reset_index()
    )
    profile["at_risk_pct"] = (
        profile["pipeline_at_risk"] / profile["total_amount"] * 100
    )
    return profile


# ── 3. PER-DIMENSION SEGMENT PROFILES (Charts 30–32) ──────────────────────
def plot_segment_profile(src, dim_col, dim_label, top_n, filename):
    """
    3-panel figure directly answering RQ4 for one dimension:
      Left  : RQ1 win-probability per segment (sorted by avg)
      Middle: RQ2 predicted days remaining per segment (sorted by urgency)
      Right : Pipeline composition — expected vs at-risk (RQ1 × amount)
    """
    prof = build_profile(src, dim_col, top_n)

    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, top_n * 0.48 + 2)))

    # ── Left: RQ1 win probability ──────────────────────────────────────────
    ax = axes[0]
    wp = prof.sort_values("avg_win_prob", ascending=True)
    bar_colors = [
        "#276749" if v >= 0.65 else ("#DD6B20" if v >= 0.33 else "#C53030")
        for v in wp["avg_win_prob"]
    ]
    ax.barh(range(len(wp)), wp["avg_win_prob"] * 100, color=bar_colors,
            edgecolor="white", linewidth=0.4)
    ax.errorbar(
        wp["avg_win_prob"] * 100, range(len(wp)),
        xerr=wp["std_win_prob"].fillna(0) * 100,
        fmt="none", color="#333", capsize=3, linewidth=0.7,
    )
    ax.set_yticks(range(len(wp)))
    ax.set_yticklabels(wp[dim_col], fontsize=8)
    ax.axvline(50, color="gray",  linestyle="--", linewidth=0.8, label="50%")
    ax.axvline(65, color="#C53030", linestyle=":", linewidth=0.8, label="65%")
    ax.set_xlabel("Avg win probability (%)")
    ax.set_title(f"RQ1 — Win Probability\nby {dim_label}", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)
    for i, row in enumerate(wp.itertuples()):
        ax.text(row.avg_win_prob * 100 + 0.3, i,
                f"n={int(row.n_deals)}", va="center", fontsize=7, color="#444")

    # ── Middle: RQ2 urgency ────────────────────────────────────────────────
    ax = axes[1]
    ur = prof.dropna(subset=["avg_days_remaining"]).sort_values(
        "avg_days_remaining", ascending=False)  # longest bar at top
    if len(ur) > 0:
        pal = sns.color_palette("YlOrRd_r", len(ur))
        ax.barh(range(len(ur)), ur["avg_days_remaining"],
                color=pal, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(ur)))
        ax.set_yticklabels(ur[dim_col], fontsize=8)
        ax.set_xlabel("Avg predicted days remaining")
        ax.set_title(f"RQ2 — Time-to-Close Urgency\nby {dim_label}"
                     f"\n(shorter bar = closes sooner)", fontsize=10)
        for i, row in enumerate(ur.itertuples()):
            ax.text(row.avg_days_remaining + 0.5, i,
                    f"{int(row.avg_days_remaining)}d  "
                    f"(med={int(row.med_days_remaining)}d)",
                    va="center", fontsize=7, color="#444")
    else:
        ax.text(0.5, 0.5, "No RQ2 data available",
                transform=ax.transAxes, ha="center")

    # ── Right: pipeline composition ────────────────────────────────────────
    ax = axes[2]
    pc = prof.sort_values("total_amount", ascending=True)
    ax.barh(range(len(pc)), pc["expected_revenue"] / 1e6,
            color="#276749", label="Expected Revenue (RQ1 × amount)")
    ax.barh(range(len(pc)), pc["pipeline_at_risk"] / 1e6,
            left=pc["expected_revenue"] / 1e6,
            color="#C53030", alpha=0.7, label="At Risk")
    ax.set_yticks(range(len(pc)))
    ax.set_yticklabels(pc[dim_col], fontsize=8)
    ax.set_xlabel("Pipeline value (M)")
    ax.set_title(f"Pipeline Health\n(Expected vs At-Risk) by {dim_label}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for i, row in enumerate(pc.itertuples()):
        total = row.total_amount / 1e6
        ax.text(total + 0.05, i,
                f"{row.at_risk_pct:.0f}% at risk",
                va="center", fontsize=7, color="#444")

    fig.suptitle(
        f"RQ4 — Model-Driven Segmentation by {dim_label}\n"
        f"Left: RQ1 win probability  |  Middle: RQ2 close urgency  |  "
        f"Right: pipeline composition",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


print("\nBuilding per-dimension segment profiles …")
plot_segment_profile(df, "business_unit", "Business Unit",  top_n=8,  filename="30_rq4_bu_model_segments.png")
plot_segment_profile(df, "org_country",   "Region",         top_n=15, filename="31_rq4_region_model_segments.png")
plot_segment_profile(df, "pricelist",     "Product Group",  top_n=12, filename="32_rq4_pg_model_segments.png")


# ── 4. STRATEGIC QUADRANT MAP (Chart 33) ──────────────────────────────────
print("\nBuilding strategic quadrant map …")

QUADRANT_LABELS = {
    (True, True):   ("Harvest Now",  "#276749"),  # high prob, short time
    (True, False):  ("Invest",       "#2B6CB0"),  # high prob, long time
    (False, True):  ("Last Chance",  "#DD6B20"),  # low prob, short time
    (False, False): ("Review/Exit",  "#C53030"),  # low prob, long time
}


def plot_quadrant_panel(src, dim_col, dim_label, top_n, ax):
    prof = build_profile(src.dropna(subset=["pred_days_remaining"]),
                         dim_col, top_n)
    if len(prof) < 3:
        ax.text(0.5, 0.5, "Insufficient data",
                transform=ax.transAxes, ha="center", va="center")
        return

    x = prof["avg_win_prob"] * 100
    y = prof["avg_days_remaining"]
    sizes = (prof["total_amount"] / prof["total_amount"].max() * 1200 + 80).clip(80, 1200)

    med_x = x.median()
    med_y = y.median()

    colors = [
        QUADRANT_LABELS[(xi >= med_x, yi <= med_y)][0]
        for xi, yi in zip(x, y)
    ]
    color_vals = [
        QUADRANT_LABELS[(xi >= med_x, yi <= med_y)][1]
        for xi, yi in zip(x, y)
    ]

    ax.scatter(x, y, s=sizes, c=color_vals, alpha=0.75,
               edgecolors="white", linewidths=0.6)

    for _, row in prof.iterrows():
        ax.annotate(
            str(row[dim_col])[:14],
            xy=(row["avg_win_prob"] * 100, row["avg_days_remaining"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=7, color="#111",
        )

    ax.axvline(med_x, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(med_y, color="gray", linestyle="--", linewidth=0.8)

    xl = x.min() - 5
    xh = x.max() + 12
    yl = max(0, y.min() - 15)
    yh = y.max() + 20
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)

    label_kw = dict(fontsize=8, alpha=0.45, ha="center", va="center",
                    fontweight="bold")
    ax.text((xl + med_x) / 2, (yl + med_y) / 2,
            "Last Chance", color="#DD6B20", **label_kw)
    ax.text((xh + med_x) / 2, (yl + med_y) / 2,
            "Harvest Now", color="#276749", **label_kw)
    ax.text((xl + med_x) / 2, (yh + med_y) / 2,
            "Review / Exit", color="#C53030", **label_kw)
    ax.text((xh + med_x) / 2, (yh + med_y) / 2,
            "Invest", color="#2B6CB0", **label_kw)

    ax.set_xlabel("Avg RQ1 Win Probability (%)", fontsize=9)
    ax.set_ylabel("Avg RQ2 Days Remaining", fontsize=9)
    ax.set_title(f"Strategic Position — {dim_label}\n"
                 f"(bubble size = total pipeline; dashed = medians)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))


fig, axes = plt.subplots(1, 3, figsize=(22, 8))
plot_quadrant_panel(df, "business_unit", "Business Unit",  top_n=8,  ax=axes[0])
plot_quadrant_panel(df, "org_country",   "Region",         top_n=15, ax=axes[1])
plot_quadrant_panel(df, "pricelist",     "Product Group",  top_n=12, ax=axes[2])

fig.suptitle(
    "RQ4 — Strategic Quadrant: RQ1 Win Probability × RQ2 Time-to-Close\n"
    "Harvest Now = high win prob + short time  |  "
    "Invest = high win prob + long time  |  "
    "Last Chance = low prob + short time  |  "
    "Review/Exit = low prob + long time",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "33_rq4_strategic_quadrant.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 33_rq4_strategic_quadrant.png")


# ── 5. WIN-PROBABILITY TIER COMPOSITION (Chart 34) ─────────────────────────
print("\nBuilding win-probability tier composition …")

TIER_COLORS = ["#276749", "#DD6B20", "#C53030"]
TIER_ORDER  = ["High (>65%)", "Medium (33–65%)", "Low (<33%)"]


def plot_tier_composition(src, dim_col, dim_label, top_n, ax):
    top_vals = src[dim_col].value_counts().head(top_n).index
    sub = src[src[dim_col].isin(top_vals)].dropna(subset=["win_tier"])

    pivot = pd.crosstab(sub[dim_col], sub["win_tier"], normalize="index") * 100
    for col in TIER_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[TIER_ORDER].sort_values("High (>65%)", ascending=True)

    pivot.plot.barh(stacked=True, ax=ax, color=TIER_COLORS,
                    edgecolor="white", linewidth=0.3, legend=(ax.get_subplotspec().colspan.start == 0))
    ax.set_xlabel("% of open deals")
    ax.set_title(f"RQ1 Win-Prob Tier Composition\nby {dim_label}", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    if ax.get_subplotspec().colspan.start == 0:
        ax.legend(title="Win-prob tier", fontsize=8, loc="lower right")


fig, axes = plt.subplots(1, 3, figsize=(22, max(7, 9)))
plot_tier_composition(df, "business_unit", "Business Unit",  top_n=8,  ax=axes[0])
plot_tier_composition(df, "org_country",   "Region",         top_n=15, ax=axes[1])
plot_tier_composition(df, "pricelist",     "Product Group",  top_n=12, ax=axes[2])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()
fig.legend(handles, labels, title="Win-prob tier (RQ1)",
           loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "RQ4 — How RQ1 Win-Probability Model Segments the Open Pipeline\n"
    "Green = High (>65%)  |  Orange = Medium (33–65%)  |  Red = Low (<33%)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "34_rq4_win_tier_composition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 34_rq4_win_tier_composition.png")


# ── 6. EXPORT SUMMARY TABLE ───────────────────────────────────────────────
print("\nExporting RQ4 segment summary …")

rows = []
for dim_col, dim_label, top_n in [
    ("business_unit", "Business Unit",  8),
    ("org_country",   "Region",         15),
    ("pricelist",     "Product Group",  12),
]:
    prof = build_profile(df, dim_col, top_n)
    prof.insert(0, "dimension", dim_label)
    prof = prof.rename(columns={dim_col: "segment"})
    rows.append(prof)

summary = pd.concat(rows, ignore_index=True)
summary["avg_win_prob_pct"]   = (summary["avg_win_prob"] * 100).round(1)
summary["expected_revenue_M"] = (summary["expected_revenue"] / 1e6).round(2)
summary["pipeline_at_risk_M"] = (summary["pipeline_at_risk"] / 1e6).round(2)
summary["avg_days_remaining"] = summary["avg_days_remaining"].round(0)

export = summary[[
    "dimension", "segment", "n_deals",
    "avg_win_prob_pct", "avg_days_remaining", "med_days_remaining",
    "expected_revenue_M", "pipeline_at_risk_M", "at_risk_pct",
]]
export.to_csv(OUT_DIR / "rq4_segment_model_summary.csv", index=False)
print("  Saved rq4_segment_model_summary.csv")


# ── 7. PRINT SUMMARY ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RQ4 — SEGMENT MODEL SUMMARY")
print("=" * 70)

for dim_col, dim_label, top_n in [
    ("business_unit", "Business Unit",  8),
    ("org_country",   "Region",         15),
    ("pricelist",     "Product Group",  12),
]:
    prof = (build_profile(df, dim_col, top_n)
            .sort_values("avg_win_prob", ascending=False))
    prof["win_prob_%"]  = (prof["avg_win_prob"] * 100).round(1)
    prof["exp_rev_M"]   = (prof["expected_revenue"] / 1e6).round(2)
    prof["at_risk_M"]   = (prof["pipeline_at_risk"] / 1e6).round(2)
    prof["days_rem"]    = prof["avg_days_remaining"].round(0)
    print(f"\n── {dim_label} (sorted by RQ1 win probability ↓) ──")
    print(prof[[dim_col, "n_deals", "win_prob_%", "days_rem",
                "exp_rev_M", "at_risk_M"]].to_string(index=False))

print("\nDone — rq4_model_segmentation.py")
