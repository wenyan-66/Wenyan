"""
Shared feature engineering for chance_changelog.parquet.
Imported by calibrated_model.py, survival_cohort.py, advanced_diagnostics.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "chance_changelog.parquet"

STAGE_RANK = {
    "Lead": 1, "Interest phase": 2, "Information Stage": 3,
    "Proposal stage": 4, "Negotiation phase": 5, "Verbal Confirmation": 6,
    "Order for registration - SC / COF": 7, "Order in revision - SC / COF": 8,
    "Order received - Sales": 9, "Draft": 2,
}

# Feature groups used by model scripts
NUMERIC_FEATURES = [
    "probability", "log_amount", "stage_rank", "deal_age",
    "days_since_last_update", "total_days_pushed", "close_date_shift",
    "saledate_delta", "prob_change_last_update", "prob_volatility",
    "amount_changes", "updates_last_14d", "updates_last_30d",
    "updates_last_180d", "update_intensity", "push_ratio",
    "rep_closing_rate", "deal_share_rep", "customer_win_ratio",
    "is_repeat_customer", "home_country", "intercompany_flag",
    "has_products_and_services", "zombie_deal",
    "prob_x_closing_rate", "stage_x_prob",
]
CAT_FEATURES = ["business_unit", "org_category_en", "chance_type_en"]
ALL_FEATURES  = NUMERIC_FEATURES + CAT_FEATURES


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    for col in ["registered_dt", "updated_dt", "saledate"]:
        df[col] = pd.to_datetime(df[col])
    return df


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw changelog → one row per opportunity.
    No filtering applied — caller decides what to include.
    """
    raw = raw.copy().sort_values(["chance_id", "updated_dt"]).reset_index(drop=True)
    g   = raw.groupby("chance_id", sort=False)

    # ── within-group deltas (vectorised) ──────────────────────────────────
    raw["prev_saledate"] = g["saledate"].shift(1)
    raw["prev_prob"]     = g["probability"].shift(1)
    raw["prev_amount"]   = g["amount"].shift(1)
    raw["prev_updated"]  = g["updated_dt"].shift(1)

    raw["saledate_chg"] = (raw["saledate"] - raw["prev_saledate"]).dt.days.fillna(0)
    raw["prob_chg"]     = (raw["probability"] - raw["prev_prob"]).fillna(0)
    raw["amount_diff"]  = (raw["amount"]      - raw["prev_amount"]).fillna(0)
    raw["days_gap"]     = (raw["updated_dt"]  - raw["prev_updated"]).dt.days.fillna(0)

    raw["max_dt"]  = g["updated_dt"].transform("max")
    raw["days_ago"] = (raw["max_dt"] - raw["updated_dt"]).dt.days.fillna(0)

    # ── deal-level aggregation ─────────────────────────────────────────────
    last_snap = g.last().reset_index()
    first_dt  = g["registered_dt"].first().reset_index(name="registered_dt")

    # Temporal aggregates computed separately for speed
    n_upd  = g.size().reset_index(name="n_updates")
    pushed = (raw[raw["saledate_chg"] > 0]
              .groupby("chance_id")["saledate_chg"].sum()
              .reset_index(name="total_days_pushed"))
    cds    = ((raw["saledate_chg"] != 0).groupby(raw["chance_id"]).sum()
              .reset_index(name="close_date_shift"))
    prob_last = g["prob_chg"].last().reset_index(name="prob_change_last_update")
    prob_vol  = g["probability"].std().reset_index(name="prob_volatility")
    amt_chg   = ((raw["amount_diff"] != 0).groupby(raw["chance_id"]).sum()
                 .reset_index(name="amount_changes"))
    days_last = g["days_gap"].last().reset_index(name="days_since_last_update")
    u14  = (raw["days_ago"] <= 14 ).groupby(raw["chance_id"]).sum().reset_index(name="updates_last_14d")
    u30  = (raw["days_ago"] <= 30 ).groupby(raw["chance_id"]).sum().reset_index(name="updates_last_30d")
    u180 = (raw["days_ago"] <= 180).groupby(raw["chance_id"]).sum().reset_index(name="updates_last_180d")
    u365 = (raw["days_ago"] <= 365).groupby(raw["chance_id"]).sum().reset_index(name="updates_last_1y")

    # Assemble (drop registered_dt from last_snap — use first-row value instead)
    last_snap = last_snap.drop(columns=["registered_dt"], errors="ignore")
    agg = (last_snap
           .merge(first_dt,    on="chance_id")
           .merge(n_upd,       on="chance_id")
           .merge(pushed,      on="chance_id", how="left")
           .merge(cds,         on="chance_id", how="left")
           .merge(prob_last,   on="chance_id", how="left")
           .merge(prob_vol,    on="chance_id", how="left")
           .merge(amt_chg,     on="chance_id", how="left")
           .merge(days_last,   on="chance_id", how="left")
           .merge(u14,         on="chance_id", how="left")
           .merge(u30,         on="chance_id", how="left")
           .merge(u180,        on="chance_id", how="left")
           .merge(u365,        on="chance_id", how="left"))

    agg = agg.rename(columns={"updated_dt": "last_updated_dt"})
    agg[["total_days_pushed", "close_date_shift"]] = (
        agg[["total_days_pushed", "close_date_shift"]].fillna(0))

    # ── derived features ───────────────────────────────────────────────────
    agg["deal_age"]       = ((agg["last_updated_dt"] - agg["registered_dt"])
                             .dt.days.clip(lower=0))
    agg["saledate_delta"] = ((agg["saledate"] - agg["registered_dt"])
                             .dt.days.clip(lower=0).fillna(0))
    agg["log_amount"]     = np.log1p(agg["amount"].clip(lower=0).fillna(0))
    agg["stage_rank"]     = agg["stage_en"].map(STAGE_RANK).fillna(3)
    agg["prob_volatility"]        = agg["prob_volatility"].fillna(0)
    agg["prob_change_last_update"]= agg["prob_change_last_update"].fillna(0)
    agg["days_since_last_update"] = agg["days_since_last_update"].fillna(0)
    agg["update_intensity"]       = (agg["updates_last_30d"]
                                     / agg["deal_age"].clip(lower=1) * 30).clip(upper=10)
    agg["push_ratio"]  = agg["total_days_pushed"] / agg["saledate_delta"].clip(lower=1)
    agg["zombie_deal"] = ((agg["deal_age"] > 365) & (agg["updates_last_180d"] <= 1)).astype(int)
    agg["year_registered"] = agg["registered_dt"].dt.year
    agg["quarter"]         = agg["registered_dt"].dt.to_period("Q").astype(str)

    # home_country
    top_country = raw["org_country"].value_counts().idxmax()
    agg["home_country"] = (agg["org_country"] == top_country).astype(int)

    # ── rep / customer historical stats ───────────────────────────────────
    closed_last = (raw[raw["status"].isin(["Won", "Lost"])]
                   .sort_values(["chance_id", "updated_dt"])
                   .groupby("chance_id").last().reset_index())
    closed_last["is_won_c"] = (closed_last["status"] == "Won").astype(int)
    global_wr = closed_last["is_won_c"].mean()

    rep_stats = (closed_last.groupby("responsible")
                 .agg(rep_closing_rate=("is_won_c","mean"),
                      rep_amount      =("amount",  "sum"))
                 .reset_index())
    cust_stats = (closed_last.groupby("org_id")
                  .agg(customer_win_ratio=("is_won_c",   "mean"),
                       n_cust_deals      =("chance_id",  "count"))
                  .reset_index())
    cust_stats["is_repeat_customer"] = (cust_stats["n_cust_deals"] > 1).astype(int)

    agg = (agg
           .merge(rep_stats,  on="responsible", how="left")
           .merge(cust_stats[["org_id","customer_win_ratio","is_repeat_customer"]],
                  on="org_id", how="left"))
    agg["rep_closing_rate"]    = agg["rep_closing_rate"].fillna(global_wr)
    agg["rep_amount"]          = agg["rep_amount"].fillna(agg["amount"])
    agg["customer_win_ratio"]  = agg["customer_win_ratio"].fillna(global_wr)
    agg["is_repeat_customer"]  = agg["is_repeat_customer"].fillna(0)

    agg["deal_share_rep"]      = (agg["amount"] / agg["rep_amount"].clip(lower=1)).clip(0, 1)
    agg["prob_x_closing_rate"] = agg["probability"] * agg["rep_closing_rate"]
    agg["stage_x_prob"]        = agg["stage_rank"]  * agg["probability"]

    # ── targets ────────────────────────────────────────────────────────────
    agg["target_won"]    = (agg["status"] == "Won").astype(int)
    agg["event_type"]    = (agg["status"]
                            .map({"Won": 1, "Lost": 2, "Interrupted": 3})
                            .fillna(0).astype(int))
    agg["is_closed"]     = (agg["event_type"] > 0).astype(int)
    agg["duration_days"] = agg["deal_age"].clip(lower=1)

    return agg


def encode_cats(df: pd.DataFrame, cat_cols=None) -> pd.DataFrame:
    """Label-encode categorical columns in place."""
    from sklearn.preprocessing import LabelEncoder
    if cat_cols is None:
        cat_cols = CAT_FEATURES
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
    return df
