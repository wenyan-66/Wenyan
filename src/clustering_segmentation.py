"""
Clustering & Segmentation Analysis on chance_changelog.parquet
==============================================================
Question: How can the model support segmentation by dimensions such as
business unit, region, and product group?

Approach:
  1. Aggregate changelog rows into one record per opportunity (chance_id),
     computing behavioural / deal metrics.
  2. Apply K-Means clustering on those metrics.
  3. Profile each cluster and show how clusters distribute across the three
     segmentation dimensions: business_unit, region (org_country), and
     product_group (pricelist).
  4. Produce a within-segment breakdown so every BU / region / product-group
     slice can be described by its cluster mix.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = Path(__file__).parents[1] / "chance_changelog.parquet"
OUT_DIR     = Path(__file__).parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS   = 5          # chosen after elbow / silhouette analysis below

# ── 1. LOAD ────────────────────────────────────────────────────────────────
print("Loading data …")
raw = pd.read_parquet(DATA_PATH)
print(f"  Raw shape: {raw.shape}")

# ── 2. AGGREGATE CHANGELOG → ONE ROW PER OPPORTUNITY ──────────────────────
print("Aggregating changelog to one row per chance_id …")

# Stage ordinal map (higher = further along the funnel)
stage_rank = {
    "Lead":                                    1,
    "Interest phase":                          2,
    "Information Stage":                       3,
    "Proposal stage":                          4,
    "Negotiation phase":                       5,
    "Verbal Confirmation":                     6,
    "Order for registration - SC / COF":       7,
    "Order in revision - SC / COF":            8,
    "Order received - Sales":                  9,
    "Draft":                                   2,   # treat as early stage
}

raw["stage_rank"] = raw["stage_en"].map(stage_rank).fillna(3)

# Status encode
status_map = {"Open": 0, "Won": 3, "Lost": 1, "Interrupted": 2}
raw["status_code"] = raw["status"].map(status_map).fillna(0)

agg = (
    raw.sort_values(["chance_id", "updated_dt"])
    .groupby("chance_id", sort=False)
    .agg(
        # dimension labels – take last (most recent) value
        business_unit      = ("business_unit",   "last"),
        org_country        = ("org_country",     "last"),
        pricelist          = ("pricelist",        "last"),
        org_business_en    = ("org_business_en", "last"),
        org_category_en    = ("org_category_en", "last"),
        chance_type_en     = ("chance_type_en",  "last"),
        # deal metrics
        final_status       = ("status",          "last"),
        final_stage_rank   = ("stage_rank",      "last"),
        max_stage_rank     = ("stage_rank",      "max"),
        final_probability  = ("probability",     "last"),
        mean_probability   = ("probability",     "mean"),
        final_amount       = ("amount",          "last"),
        max_amount         = ("amount",          "max"),
        n_updates          = ("updated_dt",      "count"),
        registered_dt      = ("registered_dt",   "first"),
        last_updated_dt    = ("updated_dt",      "last"),
        saledate           = ("saledate",        "last"),
        intercompany_flag  = ("intercompany_flag","last"),
        has_products       = ("has_products_and_services", "last"),
    )
    .reset_index()
)

# Derived time features
agg["days_active"] = (
    (agg["last_updated_dt"] - agg["registered_dt"])
    .dt.total_seconds() / 86_400
).clip(lower=0)

agg["year_registered"] = agg["registered_dt"].dt.year
agg["quarter_registered"] = agg["registered_dt"].dt.quarter

# Won / lost flag
agg["is_won"]  = (agg["final_status"] == "Won").astype(int)
agg["is_lost"] = (agg["final_status"] == "Lost").astype(int)
agg["is_open"] = (agg["final_status"] == "Open").astype(int)

# Amount log-transform (handle zeros/negatives gracefully)
agg["log_final_amount"] = np.log1p(agg["final_amount"].clip(lower=0).fillna(0))
agg["log_max_amount"]   = np.log1p(agg["max_amount"].clip(lower=0).fillna(0))

# Fill missing probability
agg["final_probability"] = agg["final_probability"].fillna(0)
agg["mean_probability"]  = agg["mean_probability"].fillna(0)

print(f"  Aggregated shape: {agg.shape}")
print(f"  Unique business_units : {agg['business_unit'].nunique()}")
print(f"  Unique regions        : {agg['org_country'].nunique()}")
print(f"  Unique pricelists     : {agg['pricelist'].nunique()}")

# ── 3. FEATURE MATRIX FOR CLUSTERING ──────────────────────────────────────
CLUSTER_FEATURES = [
    "log_final_amount",
    "log_max_amount",
    "final_probability",
    "mean_probability",
    "final_stage_rank",
    "max_stage_rank",
    "n_updates",
    "days_active",
    "intercompany_flag",
    "has_products",
    "is_won",
    "is_lost",
]

X_raw = agg[CLUSTER_FEATURES].fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 4. ELBOW & SILHOUETTE (k = 2 … 9) ─────────────────────────────────────
print("Running elbow / silhouette search …")
inertias, sil_scores = [], []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels, sample_size=10_000,
                                       random_state=RANDOM_STATE))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o", color="#2B6CB0")
axes[0].axvline(N_CLUSTERS, color="red", linestyle="--", label=f"k={N_CLUSTERS}")
axes[0].set_title("Elbow Curve (Inertia)")
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, marker="s", color="#276749")
axes[1].axvline(N_CLUSTERS, color="red", linestyle="--", label=f"k={N_CLUSTERS}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette score")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "01_elbow_silhouette.png", dpi=150)
plt.close()
print("  Saved 01_elbow_silhouette.png")

# ── 5. FINAL CLUSTERING ────────────────────────────────────────────────────
print(f"Fitting K-Means with k={N_CLUSTERS} …")
km_final = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
agg["cluster"] = km_final.fit_predict(X)

cluster_sizes = agg["cluster"].value_counts().sort_index()
print("  Cluster sizes:\n", cluster_sizes.to_string())

# ── 6. CLUSTER PROFILES ────────────────────────────────────────────────────
print("Building cluster profile …")

profile_cols = {
    "log_final_amount":   "Avg Log Deal Amount",
    "final_probability":  "Avg Final Probability",
    "final_stage_rank":   "Avg Final Stage Rank",
    "n_updates":          "Avg # Updates",
    "days_active":        "Avg Days Active",
    "is_won":             "Win Rate",
    "is_lost":            "Loss Rate",
    "is_open":            "Open Rate",
    "intercompany_flag":  "Intercompany %",
    "has_products":       "Has Products %",
}

profile = (
    agg.groupby("cluster")[list(profile_cols.keys())]
    .mean()
    .rename(columns=profile_cols)
)

# Normalise for radar/heatmap display (0–1 per column)
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

# Heatmap of normalised cluster profiles
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    profile_norm.T,
    annot=profile.T.round(2),
    fmt="g",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Normalised value (0–1)"},
)
ax.set_title("Cluster Profiles (values = raw means; colour = normalised)", fontsize=13)
ax.set_xlabel("Cluster")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_cluster_profiles.png", dpi=150)
plt.close()
print("  Saved 02_cluster_profiles.png")

# Human-readable cluster labels
cluster_labels = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}
# Derive descriptive names based on profile
win_rates = profile["Win Rate"].to_dict()
amt_ranks  = profile["Avg Log Deal Amount"].rank().to_dict()
prob_ranks = profile["Avg Final Probability"].rank().to_dict()

def describe_cluster(c):
    wr = win_rates[c]
    ar = amt_ranks[c]
    pr = prob_ranks[c]
    if wr > 0.6:
        return f"C{c}: High-Win"
    elif wr < 0.05 and ar >= 3:
        return f"C{c}: Large Open"
    elif wr < 0.05:
        return f"C{c}: Active Pipeline"
    elif ar == max(amt_ranks.values()):
        return f"C{c}: High-Value Closed"
    else:
        return f"C{c}: Mid-Range"

label_map = {c: describe_cluster(c) for c in range(N_CLUSTERS)}
agg["cluster_label"] = agg["cluster"].map(label_map)

# ── 7. SEGMENTATION BY DIMENSION ──────────────────────────────────────────
def segment_heatmap(agg_df, dim_col, dim_label, top_n, filename):
    """
    Heat-map: rows = top_n dim values, columns = clusters,
    cell = % of that segment's deals in each cluster.
    """
    # Use top_n by frequency
    top_vals = agg_df[dim_col].value_counts().head(top_n).index
    sub = agg_df[agg_df[dim_col].isin(top_vals)]

    pivot = (
        pd.crosstab(sub[dim_col], sub["cluster"], normalize="index") * 100
    )
    pivot.columns = [label_map[c] for c in pivot.columns]
    pivot.index.name = dim_label

    fig, ax = plt.subplots(figsize=(max(10, N_CLUSTERS * 2), max(6, top_n * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.4,
        ax=ax,
        vmin=0, vmax=100,
        cbar_kws={"label": "% of segment in cluster"},
    )
    ax.set_title(
        f"Cluster Distribution Within Each {dim_label}\n"
        f"(row % — shows how each {dim_label} breaks into clusters)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")
    return pivot


def segment_bar(agg_df, dim_col, dim_label, top_n, filename):
    """Stacked bar: each segment's cluster mix."""
    top_vals = agg_df[dim_col].value_counts().head(top_n).index
    sub = agg_df[agg_df[dim_col].isin(top_vals)]

    pivot = pd.crosstab(sub[dim_col], sub["cluster"], normalize="index") * 100
    pivot.columns = [label_map[c] for c in pivot.columns]

    colors = sns.color_palette("Set2", N_CLUSTERS)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(max(12, top_n * 0.6), 6),
                    color=colors, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Cluster Mix per {dim_label}", fontsize=13)
    ax.set_xlabel(dim_label)
    ax.set_ylabel("% of Opportunities")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


print("\n── Segmentation: Business Unit ──")
bu_heatmap = segment_heatmap(agg, "business_unit", "Business Unit",
                              top_n=7, filename="03a_bu_cluster_heatmap.png")
segment_bar(agg, "business_unit", "Business Unit",
            top_n=7, filename="03b_bu_cluster_bar.png")

print("\n── Segmentation: Region (org_country) ──")
region_heatmap = segment_heatmap(agg, "org_country", "Region (org_country)",
                                 top_n=20, filename="04a_region_cluster_heatmap.png")
segment_bar(agg, "org_country", "Region (org_country)",
            top_n=20, filename="04b_region_cluster_bar.png")

print("\n── Segmentation: Product Group (pricelist) ──")
pg_heatmap = segment_heatmap(agg, "pricelist", "Product Group (pricelist)",
                             top_n=15, filename="05a_productgroup_cluster_heatmap.png")
segment_bar(agg, "pricelist", "Product Group (pricelist)",
            top_n=15, filename="05b_productgroup_cluster_bar.png")

# ── 8. PCA 2D SCATTER ─────────────────────────────────────────────────────
print("\nPCA 2D projection …")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X)
agg["pca1"] = coords[:, 0]
agg["pca2"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette("Set1", N_CLUSTERS)
for c in range(N_CLUSTERS):
    mask = agg["cluster"] == c
    ax.scatter(agg.loc[mask, "pca1"], agg.loc[mask, "pca2"],
               s=4, alpha=0.3, color=palette[c], label=label_map[c])
ax.set_title(
    f"PCA 2-D Projection of Opportunities by Cluster\n"
    f"(explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance)",
    fontsize=12,
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(markerscale=3, title="Cluster")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_pca_scatter.png", dpi=150)
plt.close()
print("  Saved 06_pca_scatter.png")

# ── 9. WIN RATE BY SEGMENT × CLUSTER ──────────────────────────────────────
print("\nWin-rate heatmaps per dimension …")

def winrate_heatmap(agg_df, dim_col, dim_label, top_n, filename):
    top_vals = agg_df[dim_col].value_counts().head(top_n).index
    sub = agg_df[agg_df[dim_col].isin(top_vals)]
    pivot = sub.groupby([dim_col, "cluster"])["is_won"].mean().unstack() * 100
    pivot.columns = [label_map[c] for c in pivot.columns]
    pivot.index.name = dim_label

    fig, ax = plt.subplots(figsize=(max(10, N_CLUSTERS * 2), max(5, top_n * 0.45)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.4, ax=ax, vmin=0, vmax=100,
        cbar_kws={"label": "Win rate (%)"},
    )
    ax.set_title(f"Win Rate (%) by {dim_label} × Cluster", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")

winrate_heatmap(agg, "business_unit", "Business Unit",
                top_n=7,  filename="07a_bu_winrate.png")
winrate_heatmap(agg, "org_country",   "Region",
                top_n=20, filename="07b_region_winrate.png")
winrate_heatmap(agg, "pricelist",     "Product Group",
                top_n=15, filename="07c_productgroup_winrate.png")

# ── 10. SUMMARY TABLES ────────────────────────────────────────────────────
print("\nWriting summary CSV files …")

# (a) Overall cluster profile
profile.round(3).to_csv(OUT_DIR / "cluster_profiles.csv")

# (b) BU × cluster distribution
bu_heatmap.round(1).to_csv(OUT_DIR / "bu_cluster_distribution.csv")

# (c) Region × cluster distribution
region_heatmap.round(1).to_csv(OUT_DIR / "region_cluster_distribution.csv")

# (d) Product group × cluster distribution
pg_heatmap.round(1).to_csv(OUT_DIR / "productgroup_cluster_distribution.csv")

# (e) Per-opportunity cluster assignment (key columns only)
export_cols = [
    "chance_id", "business_unit", "org_country", "pricelist",
    "org_business_en", "org_category_en", "final_status",
    "final_amount", "final_probability", "days_active",
    "cluster", "cluster_label",
]
agg[export_cols].to_csv(OUT_DIR / "opportunity_clusters.csv", index=False)

print("  Saved cluster_profiles.csv")
print("  Saved bu_cluster_distribution.csv")
print("  Saved region_cluster_distribution.csv")
print("  Saved productgroup_cluster_distribution.csv")
print("  Saved opportunity_clusters.csv")

# ── 11. PRINT SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: Cluster profiles")
print("=" * 70)
pd.set_option("display.float_format", "{:.3f}".format)
print(profile.to_string())

print("\n" + "=" * 70)
print("SUMMARY: Cluster mix per Business Unit (row %)")
print("=" * 70)
print(bu_heatmap.round(1).to_string())

print("\n" + "=" * 70)
print("SUMMARY: Cluster mix per Region – top 10 (row %)")
print("=" * 70)
print(region_heatmap.head(10).round(1).to_string())

print("\n" + "=" * 70)
print("SUMMARY: Cluster mix per Product Group (row %)")
print("=" * 70)
print(pg_heatmap.round(1).to_string())

print("\n✓ All outputs written to:", OUT_DIR)
print("Done.")
