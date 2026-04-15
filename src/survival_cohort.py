"""
A3  Competing-Risks Survival Model   — Aalen-Johansen CIF (Won / Lost / Interrupted)
C1  Kaplan-Meier Cohort Analysis     — deal closure speed by registration quarter
                                        and by business unit

Implements KM and competing-risks estimators from scratch using numpy/scipy
(lifelines unavailable in this environment due to autograd-gamma build issue).

Outputs:
  outputs/16_km_cohort_quarter.png
  outputs/17_km_by_bu.png
  outputs/18_competing_risks_cif.png
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

from feature_engineering import load_raw, engineer_features

ROOT    = Path(__file__).parents[1]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Survival estimators ────────────────────────────────────────────────────

def kaplan_meier(durations: np.ndarray, events: np.ndarray,
                 max_t: float = None) -> pd.DataFrame:
    """
    Kaplan-Meier estimator.
    events : 1 = closed (any cause), 0 = censored (still open)
    Returns DataFrame with columns: time, survival, n_at_risk, n_events
    """
    df = pd.DataFrame({"t": durations, "e": events}).sort_values("t").reset_index(drop=True)
    times, S, n_risk, n_ev = [0], [1.0], [len(df)], [0]
    S_cur = 1.0
    n     = len(df)

    for t, grp in df[df["e"] == 1].groupby("t"):
        if max_t and t > max_t:
            break
        d = len(grp)                            # events at t
        r = (df["t"] >= t).sum()                # at risk at t
        S_cur *= (1 - d / r)
        times.append(t);  S.append(S_cur)
        n_risk.append(r); n_ev.append(d)

    return pd.DataFrame({"time": times, "survival": S,
                          "n_at_risk": n_risk, "n_events": n_ev})


def aalen_johansen(durations: np.ndarray, event_types: np.ndarray,
                   cause: int) -> pd.DataFrame:
    """
    Aalen-Johansen estimator for the cumulative incidence function (CIF)
    of one competing event type.

    event_types : 0=censored, 1=Won, 2=Lost, 3=Interrupted
    cause       : which event type to compute CIF for
    Returns DataFrame with columns: time, cif
    """
    df = pd.DataFrame({"t": durations, "e": event_types}).sort_values("t").reset_index(drop=True)
    n  = len(df)

    # Overall KM survival (treating any non-censored event as event)
    any_event = (df["e"] > 0).astype(int)
    km = kaplan_meier(df["t"].values, any_event.values)
    km = km.set_index("time")["survival"].to_dict()

    # CIF for the target cause
    times = [0]
    cif   = [0.0]
    cif_val = 0.0
    event_times = df[df["e"] == cause]["t"].values

    for t in sorted(np.unique(event_times)):
        r   = (df["t"] >= t).sum()              # at risk
        d_j = ((df["t"] == t) & (df["e"] == cause)).sum()   # cause events

        # S(t-) from overall KM
        prev_times = [k for k in km if k < t]
        S_prev = km[max(prev_times)] if prev_times else 1.0

        cif_val += (d_j / r) * S_prev
        times.append(t)
        cif.append(cif_val)

    return pd.DataFrame({"time": times, "cif": cif})


# ── 1. LOAD & ENGINEER ─────────────────────────────────────────────────────
print("Loading data …")
raw = load_raw()
df  = engineer_features(raw)

# Caps to avoid extreme outliers in plots
MAX_DAYS = 1000
df["duration_plot"] = df["duration_days"].clip(upper=MAX_DAYS)
df["any_closure"]   = (df["is_closed"] == 1).astype(int)

# ── 2. KM BY REGISTRATION QUARTER (C1) ────────────────────────────────────
print("Kaplan-Meier by registration quarter …")
quarters = sorted(df["quarter"].unique())
# Keep quarters with ≥ 50 deals and at least 1 closure
valid_q = [q for q in quarters
           if len(df[df["quarter"] == q]) >= 50
           and df.loc[df["quarter"] == q, "any_closure"].sum() > 0]

palette = sns.color_palette("viridis", len(valid_q))
fig, ax = plt.subplots(figsize=(12, 6))

for q, color in zip(valid_q, palette):
    sub = df[df["quarter"] == q]
    km  = kaplan_meier(sub["duration_plot"].values, sub["any_closure"].values)
    ax.step(km["time"], km["survival"], where="post",
            color=color, linewidth=1.2, alpha=0.85, label=q)

ax.set_xlabel("Days since registration")
ax.set_ylabel("P(deal still open)")
ax.set_title("C1 — Kaplan-Meier: Deal Closure Speed by Registration Quarter\n"
             "Steeper drop = cohort closes faster; flat = stalling deals", fontsize=11)
ax.legend(title="Quarter", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=7, ncol=2)
ax.set_xlim(0, MAX_DAYS)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "16_km_cohort_quarter.png", dpi=150)
plt.close(fig)
print("  Saved 16_km_cohort_quarter.png")

# ── 3. KM BY BUSINESS UNIT (C1 extension) ─────────────────────────────────
print("Kaplan-Meier by business unit …")
bus = df["business_unit"].value_counts().index.tolist()[:7]
pal = sns.color_palette("Set1", len(bus))

fig, ax = plt.subplots(figsize=(11, 6))
for bu, color in zip(bus, pal):
    sub = df[df["business_unit"] == bu]
    km  = kaplan_meier(sub["duration_plot"].values, sub["any_closure"].values)
    med = km.loc[km["survival"] <= 0.5, "time"]
    median_label = f"  (med={med.iloc[0]:.0f}d)" if len(med) > 0 else ""
    ax.step(km["time"], km["survival"], where="post",
            color=color, linewidth=1.8, label=f"{bu[:12]}{median_label}")

ax.set_xlabel("Days since registration")
ax.set_ylabel("P(deal still open)")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
ax.set_title("C1 — Kaplan-Meier: Deal Closure Speed by Business Unit\n"
             "Dashed line = median closure time", fontsize=11)
ax.legend(title="Business Unit", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.set_xlim(0, MAX_DAYS)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "17_km_by_bu.png", dpi=150)
plt.close(fig)
print("  Saved 17_km_by_bu.png")

# ── 4. COMPETING RISKS — AALEN-JOHANSEN (A3) ──────────────────────────────
print("Competing risks (Aalen-Johansen CIF) …")

dur  = df["duration_plot"].values
evts = df["event_type"].values    # 0=open, 1=Won, 2=Lost, 3=Interrupted

cause_labels = {1: "Won", 2: "Lost", 3: "Interrupted"}
cause_colors = {1: "#276749", 2: "#C53030", 3: "#744210"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: overall CIF
ax = axes[0]
cif_all = {}
for cause, label in cause_labels.items():
    cif = aalen_johansen(dur, evts, cause)
    cif_all[label] = cif
    ax.step(cif["time"], cif["cif"], where="post",
            color=cause_colors[cause], linewidth=2, label=label)

ax.set_xlabel("Days since registration")
ax.set_ylabel("Cumulative incidence (probability)")
ax.set_title("A3 — Competing Risks: Cumulative Incidence Function\n"
             "Each curve = P(event of that type by day t)\n"
             "Sum of all curves = overall closure rate", fontsize=10)
ax.legend()
ax.set_xlim(0, MAX_DAYS)
ax.grid(alpha=0.25)

# Right: CIF by business unit for Won only
ax = axes[1]
for bu, color in zip(bus[:5], pal[:5]):
    sub_mask = df["business_unit"] == bu
    cif = aalen_johansen(dur[sub_mask], evts[sub_mask], cause=1)  # Won
    ax.step(cif["time"], cif["cif"], where="post",
            color=color, linewidth=1.6, label=bu[:12])

ax.set_xlabel("Days since registration")
ax.set_ylabel("P(Won by day t)")
ax.set_title("A3 — Win CIF by Business Unit\n"
             "Higher plateau = higher eventual win rate", fontsize=10)
ax.legend(title="Business Unit", fontsize=8)
ax.set_xlim(0, MAX_DAYS)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_DIR / "18_competing_risks_cif.png", dpi=150)
plt.close(fig)
print("  Saved 18_competing_risks_cif.png")

# ── 5. PRINT SUMMARY ──────────────────────────────────────────────────────
print("\n── Competing-risks summary (probability at day 365) ──")
for label, cif_df in cif_all.items():
    at365 = cif_df.loc[cif_df["time"] <= 365, "cif"]
    prob  = at365.iloc[-1] if len(at365) > 0 else 0
    print(f"  P({label} by day 365) = {prob:.3f}")

print("\nDone — survival_cohort.py")
