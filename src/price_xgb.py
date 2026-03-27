"""
VR Price Prediction — XGBoost Regression
=========================================

Predicts Sales Price (USD per unit) for Voltage Regulator quotes.
Reuses the prepared train/test data from the Transfer project as-is.

Three models are trained separately:
  all  — all quotes combined (Won + Lost)
  won  — won deals only  -> learns accepted / market-clearing prices
  lost — lost deals only -> learns rejected / above-market prices

Comparing all three outputs for a new quote tells you where your price
sits relative to the market and whether you are above or below typical
winning levels.

----------------------------------------------------------------------
COLUMNS EXCLUDED — DATA LEAKAGE
  Sales Price                    the target itself
  Quote Line Item: Total Price   = Sales Price x Order Quantity
  price_per_kVA                  = Sales Price / kVA
  log_sales_price                = log(Sales Price)
  log_total_price                = log(Total Price)

COLUMNS EXCLUDED — NEW-CUSTOMER GENERALISABILITY
  Account Name_freq              encodes how often this customer appears
  End User Customer_freq         same for end-user
  Account Name                   customer identity
  End User Customer              customer identity
----------------------------------------------------------------------

Usage:
    python src/price_xgb.py                        # train 'all' subset
    python src/price_xgb.py --subset won           # won deals only
    python src/price_xgb.py --subset lost          # lost deals only
    python src/price_xgb.py --subset all_subsets   # train all three

Output (per subset):
    models/<subset>/price_model_xgb.joblib         trained XGB pipeline
    models/<subset>/metrics.json                   evaluation metrics
    models/<subset>/ebm_feature_importance.json    EBM global importances
    models/<subset>/ebm_feature_importance.png     importance bar chart
    models/<subset>/ebm_model.joblib               fitted EBM model
    models/price_prediction_report.txt             comparison table (all_subsets)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Reuse the prepared data from the Transfer project — no re-preparation needed
DATA_DIR = Path("xxxxx")


# =============================================================================
# Feature Definitions
# =============================================================================
# These lists define exactly which columns feed into the model.
# Adding or removing a feature is as simple as editing the list here.

TARGET_COL = "Sales Price"

# Numeric features — XGBoost does not require scaling, so we only impute medians
NUMERIC_FEATURES = [
    "Order Quantity",
    "kVA_numeric",
    "voltage_kV_actual",
    "Mounting_binary",
    "is_three_phase",
    "is_ONAF",
    "is_padmount_pn",
    "IsDD_Product",
    "Is_Oven",
    "Stainless Steel Tank",
    "ReRated",
    "total_kVA",
    "qty_bin",
    "Shipping State/Province_freq",   # regional market activity — not customer-specific
]

# Binary flags extracted from the free-text description column.
# These are already 0/1 integers, so they pass through without transformation.
DESC_FEATURES = [
    "desc_type_A", "desc_type_B",
    "desc_MJ4A", "desc_SEL", "desc_Beckwith", "desc_no_control",
    "desc_RS232", "desc_fiber", "desc_has_comms",
    "desc_pad", "desc_stands", "desc_NEMA_pad", "desc_rack",
    "desc_SS_tank", "desc_SS_CB", "desc_FR3", "desc_all_copper",
    "desc_lightning", "desc_bushing_guard", "desc_HV_bushings",
    "desc_RCT", "desc_aux_PT", "desc_heater",
    "desc_50Hz", "desc_rerated",
    "accessory_count",
]

# Categorical features — one-hot encoded.
# Unknown values seen at inference time are silently ignored.
CATEGORICAL_FEATURES = [
    "SE Region Name",
    "Project Execution Type",
    "Country of Installation",
    "Shipping State/Province",
]

# XGBoost model settings
XGB_PARAMS = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,          # L1 regularisation — reduces overfitting
    "objective":       "reg:squarederror",
    "random_state":    42,
    "n_jobs":          -1,           # use all CPU cores
    "verbosity":       0,
}

# EBM model settings
EBM_PARAMS = {
    "max_bins":        256,
    "max_interaction_bins": 32,
    "interactions":    10,           # top-10 pairwise interactions
    "learning_rate":   0.01,
    "max_rounds":      5000,
    "min_samples_leaf": 2,
    "random_state":    42,
    "n_jobs":          -1,
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: Path):
    """
    Load the prepared train and test sets.
    Looks for parquet first (faster), falls back to CSV.
    """
    train_path = data_dir / "train.parquet"
    test_path  = data_dir / "test.parquet"

    if not train_path.exists():
        train_path = data_dir / "train.csv"
        test_path  = data_dir / "test.csv"

    if not train_path.exists():
        print(f"ERROR: No prepared data found in: {data_dir}")
        print("Run prepare_data.py in the Transfer project first.")
        sys.exit(1)

    reader = pd.read_parquet if train_path.suffix == ".parquet" else pd.read_csv
    train = reader(train_path)
    test  = reader(test_path)

    print(f"  train: {len(train)} rows   test: {len(test)} rows")
    return train, test


# =============================================================================
# Subset Filtering
# =============================================================================

def filter_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """
    Return only the rows relevant for a given training subset.

    The 'target' column was created by prepare_data.py:
        1 = Won (Closed Booked / Order Pending)
        0 = Lost (all other kept stages)
    """
    if subset == "all":
        return df.copy()
    elif subset == "won":
        return df[df["target"] == 1].copy()
    elif subset == "lost":
        return df[df["target"] == 0].copy()
    else:
        raise ValueError(f"Unknown subset '{subset}'. Use: all | won | lost")


# =============================================================================
# Feature Selection Helper
# =============================================================================

def select_available(df: pd.DataFrame, wanted: list) -> list:
    """
    Return only the columns from 'wanted' that actually exist in df.
    Prints a warning for any that are missing so nothing fails silently.
    """
    available = [c for c in wanted if c in df.columns]
    missing   = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} requested columns not in data: {missing}")
    return available


# =============================================================================
# Pipeline Construction
# =============================================================================

def build_pipeline(numeric_cols: list, desc_cols: list, cat_cols: list) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessing + XGBoost regressor.

    ColumnTransformer routes each column group through its own preprocessing:
      numeric  -> fill missing values with the median
      desc     -> pass through unchanged (already 0/1)
      cat      -> fill missing with 'Unknown', then one-hot encode

    Columns not in any group are dropped (remainder='drop'), which is how
    the leaking columns (Sales Price, Total Price, etc.) are excluded —
    they simply never appear in numeric_cols, desc_cols, or cat_cols.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            ),
            (
                "desc",
                "passthrough",
                desc_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",   # anything not listed above is silently ignored
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**XGB_PARAMS)),
    ])


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(subset: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Generate predictions and compute regression metrics.

    Metrics explained:
      RMSE  — typical prediction error in USD (sensitive to large errors)
      MAE   — average absolute error in USD (more robust than RMSE)
      R²    — how much price variance the model explains (1.0 = perfect)
      MAPE  — average error as a percentage of the actual price
    """
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred    # positive = model under-predicted

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # MAPE: skip rows where actual = 0 to avoid division by zero
    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask].values - y_pred[mask]) / y_test[mask].values)) * 100)

    metrics = {
        "subset":          subset,
        "n_test":          int(len(y_test)),
        "rmse_usd":        round(rmse, 2),
        "mae_usd":         round(mae, 2),
        "r2":              round(r2, 4),
        "mape_pct":        round(mape, 2),
        "residual_mean":   round(float(residuals.mean()), 2),
        "residual_std":    round(float(residuals.std()), 2),
        "residual_p10":    round(float(np.percentile(residuals, 10)), 2),
        "residual_p90":    round(float(np.percentile(residuals, 90)), 2),
        "actual_mean_usd": round(float(y_test.mean()), 2),
        "actual_median_usd": round(float(y_test.median()), 2),
    }

    print(f"\n  Evaluation — subset: {subset}")
    print(f"    Test rows :      {metrics['n_test']}")
    print(f"    RMSE       :     ${rmse:>10,.0f}")
    print(f"    MAE        :     ${mae:>10,.0f}")
    print(f"    R2         :     {r2:>10.4f}")
    print(f"    MAPE       :     {mape:>10.1f}%")
    print(f"    Actual mean:     ${y_test.mean():>10,.0f}")
    print(f"    Actual median:   ${y_test.median():>10,.0f}")
    print(f"    Residuals p10/p90: ${np.percentile(residuals, 10):,.0f} / ${np.percentile(residuals, 90):,.0f}")

    return metrics


# =============================================================================
# EBM Feature Importance Analysis
# =============================================================================

def _get_ebm_feature_names(
    numeric_cols: list,
    desc_cols: list,
    cat_cols: list,
    cat_encoder: OneHotEncoder,
) -> list:
    """
    Reconstruct the flat list of feature names in the same order that the
    ColumnTransformer produces them:
        [ numeric_cols... | desc_cols... | OHE-expanded cat columns... ]
    """
    ohe_names = list(cat_encoder.get_feature_names_out(cat_cols))
    return numeric_cols + desc_cols + ohe_names


def run_ebm_analysis(
    subset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list,
    desc_cols: list,
    cat_cols: list,
    xgb_pipeline: Pipeline,
    subset_dir: Path,
    top_n: int = 30,
):
    """
    Fit an Explainable Boosting Machine (EBM) on the *same preprocessed*
    features used by the XGBoost model, then:

      1. Extract global feature importances (mean absolute score per feature).
      2. Save the ranked importances to JSON.
      3. Plot the top-N features as a horizontal bar chart (PNG).
      4. Persist the fitted EBM model for later local explanations.

    Why EBM instead of SHAP on XGBoost?
    ------------------------------------
    EBM is a glass-box model: its feature scores are exact, not approximated.
    Training a separate EBM on the same features gives a model-agnostic,
    highly interpretable importance ranking that is not subject to the
    approximation error of SHAP TreeExplainer or permutation importance.
    The XGBoost model remains the production predictor; the EBM is used
    purely for explanation.

    Parameters
    ----------
    subset        : name of the current subset ("all" | "won" | "lost")
    train_df      : raw training DataFrame (pre-pipeline)
    test_df       : raw test DataFrame (pre-pipeline)
    numeric_cols  : numeric feature column names
    desc_cols     : binary description flag column names
    cat_cols      : categorical feature column names
    xgb_pipeline  : the already-fitted XGBoost pipeline (used to extract
                    the fitted OneHotEncoder for feature name reconstruction)
    subset_dir    : output directory (models/<subset>/)
    top_n         : how many top features to show in the bar chart
    """
    print(f"\n  --- EBM Feature Importance Analysis (subset: {subset}) ---")

    # ------------------------------------------------------------------
    # Step 1: Extract the preprocessed (numeric) matrix that the pipeline
    # already knows how to build — reuse the fitted ColumnTransformer so
    # the EBM sees exactly the same transformed data as XGBoost.
    # ------------------------------------------------------------------
    preprocessor = xgb_pipeline.named_steps["preprocessor"]

    train_subset = filter_subset(train_df, subset)
    test_subset  = filter_subset(test_df, subset)

    y_train = train_subset[TARGET_COL].values
    y_test  = test_subset[TARGET_COL].values

    X_train_proc = preprocessor.transform(train_subset)
    X_test_proc  = preprocessor.transform(test_subset)

    # ------------------------------------------------------------------
    # Step 2: Reconstruct human-readable feature names
    # ------------------------------------------------------------------
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe             = cat_transformer.named_steps["encoder"]
    feature_names   = _get_ebm_feature_names(numeric_cols, desc_cols, cat_cols, ohe)

    # Guard against shape mismatch (e.g. if some OHE categories are absent)
    if X_train_proc.shape[1] != len(feature_names):
        print(
            f"  WARNING: feature name count ({len(feature_names)}) != "
            f"matrix width ({X_train_proc.shape[1]}). "
            "Using generic names."
        )
        feature_names = [f"f{i}" for i in range(X_train_proc.shape[1])]

    # ------------------------------------------------------------------
    # Step 3: Convert to DataFrame so EBM stores column names internally
    # ------------------------------------------------------------------
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names)

    # ------------------------------------------------------------------
    # Step 4: Fit EBM
    # ------------------------------------------------------------------
    print(f"  Training EBM on {len(X_train_df)} rows, {X_train_df.shape[1]} features...")
    ebm = ExplainableBoostingRegressor(**EBM_PARAMS)
    ebm.fit(X_train_df, y_train)
    print("  EBM training complete.")

    # ------------------------------------------------------------------
    # Step 5: Extract global importances
    # EBM stores per-term importances in ebm_global.data(i)["scores"].
    # The mean absolute score over the feature's bins is the standard
    # importance measure used by InterpretML.
    # ------------------------------------------------------------------
    ebm_global     = ebm.explain_global(name=f"EBM Global — {subset}")
    n_terms        = len(ebm_global.data()["names"])

    importances = []
    for i in range(n_terms):
        term_name   = ebm_global.data()["names"][i]
        term_score  = ebm_global.data()["scores"][i]   # scalar importance value
        importances.append({"feature": term_name, "importance": float(term_score)})

    # Sort descending by importance
    importances.sort(key=lambda x: x["importance"], reverse=True)

    # ------------------------------------------------------------------
    # Step 6: Save JSON
    # ------------------------------------------------------------------
    json_path = subset_dir / "ebm_feature_importance.json"
    json_path.write_text(json.dumps(importances, indent=2), encoding="utf-8")
    print(f"  EBM importances saved: {json_path}")

    # ------------------------------------------------------------------
    # Step 7: Plot top-N bar chart
    # ------------------------------------------------------------------
    top = importances[:top_n]
    names  = [d["feature"]    for d in top]
    scores = [d["importance"] for d in top]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(len(names)), scores[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean Absolute EBM Score (feature importance)", fontsize=10)
    ax.set_title(
        f"EBM Feature Importance — subset: {subset}\n"
        f"(top {len(top)} of {n_terms} terms)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = subset_dir / "ebm_feature_importance.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  EBM importance plot saved: {plot_path}")

    # ------------------------------------------------------------------
    # Step 8: Evaluate EBM on test set and print a quick comparison
    # ------------------------------------------------------------------
    y_pred_ebm = ebm.predict(X_test_df)
    ebm_rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred_ebm)))
    ebm_r2     = float(r2_score(y_test, y_pred_ebm))
    print(f"  EBM test RMSE: ${ebm_rmse:,.0f}   R²: {ebm_r2:.4f}  "
          f"(reference only — XGBoost is the production model)")

    # ------------------------------------------------------------------
    # Step 9: Save fitted EBM for local / interactive explanations later
    # ------------------------------------------------------------------
    ebm_model_path = subset_dir / "ebm_model.joblib"
    joblib.dump(ebm, ebm_model_path)
    print(f"  EBM model saved: {ebm_model_path}")

    # Print top-10 to console for quick review
    print(f"\n  Top 10 features by EBM importance (subset: {subset}):")
    print(f"  {'Rank':<5} {'Feature':<45} {'Importance':>12}")
    print(f"  {'-'*65}")
    for rank, entry in enumerate(importances[:10], start=1):
        print(f"  {rank:<5} {entry['feature']:<45} {entry['importance']:>12.4f}")

    return importances


# =============================================================================
# Report
# =============================================================================

def save_report(all_metrics: list, output_path: Path):
    """Write a human-readable comparison table for all subsets."""
    lines = [
        "=" * 70,
        "VOLTAGE REGULATOR — PRICE PREDICTION REPORT (XGBoost)",
        "=" * 70,
        "",
        f"{'Subset':<10} {'N test':>7} {'RMSE':>11} {'MAE':>11} {'R2':>7} {'MAPE':>8}",
        "-" * 60,
    ]
    for m in all_metrics:
        lines.append(
            f"{m['subset']:<10} {m['n_test']:>7} "
            f"${m['rmse_usd']:>10,.0f} ${m['mae_usd']:>10,.0f} "
            f"{m['r2']:>7.4f} {m['mape_pct']:>7.1f}%"
        )
    lines += [
        "",
        "INTERPRETING THE THREE SUBSETS",
        "-" * 40,
        "  all  : trained on all quotes — general price level predictor",
        "  won  : trained on won deals  — predicts market-clearing price",
        "  lost : trained on lost deals — predicts above-market price",
        "",
        "For a new quote:",
        "  If your price is close to 'won' prediction -> competitively priced",
        "  If your price is close to 'lost' prediction -> likely overpriced",
    ]

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"\nReport saved: {output_path}")
    print("\n" + text)


# =============================================================================
# Train One Subset
# =============================================================================

def train_subset(
    subset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list,
    desc_cols: list,
    cat_cols: list,
    models_dir: Path,
):
    """
    Full training loop for one subset:
      1. Filter rows
      2. Separate features from target
      3. Build and fit the XGBoost pipeline
      4. Evaluate on the test set
      5. Save model + metrics to models/<subset>/
      6. Run EBM feature importance analysis
    """
    print(f"\n{'='*60}")
    print(f"SUBSET: {subset.upper()}")
    print(f"{'='*60}")

    train = filter_subset(train_df, subset)
    test  = filter_subset(test_df, subset)
    print(f"  Rows — train: {len(train)} | test: {len(test)}")

    if len(train) < 10:
        print(f"  SKIP: too few rows for subset '{subset}' (need at least 10)")
        return None

    y_train = train[TARGET_COL]
    y_test  = test[TARGET_COL]

    print(
        f"  Target range — "
        f"min: ${y_train.min():,.0f}  "
        f"median: ${y_train.median():,.0f}  "
        f"max: ${y_train.max():,.0f}"
    )

    # ------------------------------------------------------------------
    # Build pipeline and train XGBoost
    # ------------------------------------------------------------------
    pipeline = build_pipeline(numeric_cols, desc_cols, cat_cols)
    print(f"\n  Training XGBoost ({XGB_PARAMS['n_estimators']} trees)...")

    # Note: the full DataFrame is passed in — the ColumnTransformer inside the
    # pipeline will pick only the columns it knows about and drop everything else.
    # This is how leaking columns are excluded without pre-filtering the DataFrame.
    pipeline.fit(train, y_train)
    print("  Done.")

    # ------------------------------------------------------------------
    # Evaluate XGBoost
    # ------------------------------------------------------------------
    metrics = evaluate(subset, pipeline, test, y_test)

    # ------------------------------------------------------------------
    # Persist XGBoost model + metrics
    # ------------------------------------------------------------------
    subset_dir = models_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    model_path   = subset_dir / "price_model_xgb.joblib"
    metrics_path = subset_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  Model   saved: {model_path}")
    print(f"  Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # EBM feature importance analysis
    # ------------------------------------------------------------------
    run_ebm_analysis(
        subset=subset,
        train_df=train_df,
        test_df=test_df,
        numeric_cols=numeric_cols,
        desc_cols=desc_cols,
        cat_cols=cat_cols,
        xgb_pipeline=pipeline,
        subset_dir=subset_dir,
    )

    return metrics


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost price regression model for Voltage Regulator quotes"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all_subsets",
        choices=["all", "won", "lost", "all_subsets"],
        help=(
            "Which data subset to train on.\n"
            "  all         = all quotes combined\n"
            "  won         = won deals only\n"
            "  lost        = lost deals only\n"
            "  all_subsets = train all three (default)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing train.csv / train.parquet",
    )
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    models_dir = PROJECT_ROOT / "models"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading prepared data...")
    train_df, test_df = load_data(data_dir)

    # Validate target column
    if TARGET_COL not in train_df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found in the data.")
        print(f"Columns available: {sorted(train_df.columns.tolist())}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve feature columns
    # Only use columns that are actually present in the data.
    # Missing columns produce a warning, not a crash.
    # ------------------------------------------------------------------
    numeric_cols = select_available(train_df, NUMERIC_FEATURES)
    desc_cols    = select_available(train_df, DESC_FEATURES)
    cat_cols     = select_available(train_df, CATEGORICAL_FEATURES)

    print(f"\nFeatures used: {len(numeric_cols)} numeric  |  "
          f"{len(desc_cols)} description flags  |  "
          f"{len(cat_cols)} categorical")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    subsets_to_run = ["all", "won", "lost"] if args.subset == "all_subsets" else [args.subset]

    all_metrics = []
    for subset in subsets_to_run:
        metrics = train_subset(
            subset, train_df, test_df,
            numeric_cols, desc_cols, cat_cols,
            models_dir,
        )
        if metrics:
            all_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Summary report (only when multiple subsets were run)
    # ------------------------------------------------------------------
    if len(all_metrics) > 1:
        report_path = models_dir / "price_prediction_report.txt"
        save_report(all_metrics, report_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
