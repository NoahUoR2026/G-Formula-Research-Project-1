'''
CAUTI PROBABILITY MODEL + G-FORMULA SIMULATION
(Day + Stability + Hybrid + CAUTI-RISK-THRESHOLD policies)
'''


import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV



# ==============================================================================
# PART 1 - BUILD THE CAUTI PROBABILITY MODEL
# ==============================================================================


DATA_PATH = "Dataset.csv"   # authoritative source - the CSV has fewer columns
DRY_RUN = False              # True = structural checks only, skip model training

TARGET = "cauti_in_period"
STAY_KEY = ["subject_id", "hadm_id", "stay_id"]
RUN_KEY = STAY_KEY + ["cath_run"]

# Fallback if the panel window cannot be measured (see Part 0, step 5)
DEFAULT_WINDOW_HOURS = 48.0


PREDICTORS = [
    "age",
    "sex_M",
    "episode_index",
    "interval_hours",
    "periods_in_state",
    "itemid_229371__last",      # Bladder Scan Estimate
    "itemid_220739__last",      # GCS Eye
    "itemid_223901__last",      # GCS Motor
    "itemid_223900__last",      # GCS Verbal
    "itemid_220045__mean",      # Heart Rate
    "itemid_220052__mean",      # Arterial Blood Pressure Mean
    "itemid_220210__mean",      # Respiratory Rate
    "itemid_223762__mean",      # Temperature (Celsius)
    "itemid_220615__last",      # Creatinine
    "itemid_225624__last",      # BUN
    "itemid_227471__last",      # Urine Specific Gravity
    "itemid_220546__last",      # White Blood Cell count
    "itemid_229357__last",      # Absolute Neutrophils
    "itemid_220228__last",      # Haemoglobin
    "itemid_227457__last",      # Platelet Count
    "itemid_220645__last",      # Sodium
    "itemid_227442__last",      # Potassium
    "itemid_220602__last",      # Chloride
    "itemid_220621__last",      # Glucose
    "itemid_227456__last",      # Albumin
    "itemid_220644__last",      # ALT
    "itemid_220587__last",      # AST
    "itemid_225690__last",      # Total Bilirubin
    "itemid_225612__last",      # Alkaline Phosphatase
    "itemid_227073__last",      # Anion Gap
    "itemid_220235__last",      # Arterial CO2 Pressure
    "itemid_220224__last",      # Arterial O2 Pressure
    "itemid_220227__last",      # Arterial O2 Saturation
    "itemid_220051__mean",      # Arterial BP Diastolic
    "itemid_220050__mean",      # Arterial BP Systolic
]

EXTRA_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "episode_index",
    "periods_in_state",
    "interval_hours",
    "removed_in_period",
    "reinsertion_in_period",
    "at_risk_cauti",            # catheter in place OR inside the panel's own window
    "at_risk_reinsertion",      # catheter out -> used to identify catheter-in rows
    "cauti_in_period",
    "split",
    # Stability trigger only, not model predictors.
    "itemid_223762__last",      # Temperature (C)
    "itemid_220045__last",      # Heart Rate
    "itemid_220050__last",      # Arterial BP Systolic  (42.8% present)
    "itemid_220179__last",      # NBP Systolic          (70.3% present)
    "itemid_220227__last",      # Arterial O2 Sat       (13.7% present)
    "itemid_220277__last",      # O2 Sat, pulse ox      (94.4% present)
]

# --- 1. Sort into time order -------------------------------------------------
# Sorting on stay first, then episode_index. episode_index is a per-period row
# counter, so within a stay it is the closest thing to a timestamp available.
# periods_in_state is deliberately NOT a sort key: it resets on every state
# change, so using it reorders rows across state boundaries.
df_full = df_full.sort_values(
    ["subject_id", "hadm_id", "stay_id", "episode_index"]
).reset_index(drop=True)
print("Data sorted by stay then episode_index.")


# --- 2. Catheter state -------------------------------------------------------
# The diagnostics showed 18,112 rows flagged at risk of BOTH CAUTI and
# reinsertion, and zero rows flagged at risk of neither. A patient can only be
# reinsertion-eligible once the catheter is out, so at_risk_cauti spans
# catheterised rows AND the panel's own post-removal window. That makes
# at_risk_reinsertion the clean state indicator.
df_full["cath_in_observed"] = (df_full["at_risk_reinsertion"] == 0).astype(int)
df_full["cath_out_observed"] = 1 - df_full["cath_in_observed"]

n_in = int(df_full["cath_in_observed"].sum())
n_out = int(df_full["cath_out_observed"].sum())
n_neither = int(((df_full["at_risk_cauti"] == 0) & (df_full["at_risk_reinsertion"] == 0)).sum())

print("\n" + "-" * 70)
print("STRUCTURAL CHECKS")
print("-" * 70)
print("Catheterised rows (at_risk_reinsertion == 0):", n_in)
print("Catheter-out rows:                           ", n_out)
print("Rows at risk of neither (want 0):            ", n_neither)

if n_neither > 0:
    print("  WARNING: some rows are at risk of neither outcome, so at_risk_reinsertion == 0")
    print("  no longer implies the catheter is in. Load state_is_out and use that instead.")

# Every recorded removal must sit on a catheterised row
rem_off = int(((df_full["removed_in_period"] == 1) & (df_full["cath_in_observed"] == 0)).sum())
print("Removals on non-catheterised rows (want 0):  ", rem_off)


# --- 3. Catheter-in runs ------------------------------------------------------
# A run opens when the catheter goes out -> in. Out-of-catheter rows inherit the
# preceding run id, so one run = one insertion, its removal, and the periods
# that follow it. episode_index is not used for grouping anywhere.
in_flag = df_full["cath_in_observed"] == 1
prev_in = (
    in_flag.groupby([df_full[c] for c in STAY_KEY], sort=False)
    .shift(1).fillna(False).astype(bool)
)
run_start = (in_flag & ~prev_in).astype(int)
df_full["cath_run"] = run_start.groupby([df_full[c] for c in STAY_KEY], sort=False).cumsum()

n_stays = df_full.groupby(STAY_KEY, sort=False).ngroups
n_reins = int((df_full["reinsertion_in_period"] == 1).sum())
n_removals = int((df_full["removed_in_period"] == 1).sum())
n_runs = int(run_start.sum())

print("\nUnique ICU stays:                  ", n_stays)
print("Reinsertion events:                ", n_reins)
print("Expected insertions (stays+reins): ", n_stays + n_reins)
print("Catheter-in runs detected:         ", n_runs)
print("Removal events:                    ", n_removals)

if abs(n_runs - (n_stays + n_reins)) > 0.05 * (n_stays + n_reins):
    print("  WARNING: run count does not match the expected insertion count.")
    print("  Do not trust the per-run results until this is reconciled.")


# --- 4. Within-run day counter -----------------------------------------------
# Policy triggers need days since THIS insertion. periods_in_state should supply
# that, but the diagnostics showed non-monotonic sequences within a stay, so a
# positional counter over catheterised rows is used instead and the agreement
# with periods_in_state is reported. Low agreement means the row ordering is
# unreliable and every time-based result in this script is suspect.
in_rows_mask = df_full["cath_in_observed"] == 1
df_full["policy_day"] = np.nan
df_full.loc[in_rows_mask, "policy_day"] = (
    df_full.loc[in_rows_mask].groupby(RUN_KEY, sort=False).cumcount() + 1
)

agree = (
    df_full.loc[in_rows_mask, "policy_day"]
    == df_full.loc[in_rows_mask, "periods_in_state"]
).mean()
print("\npolicy_day agrees with periods_in_state on %.1f %% of catheterised rows"
      % (float(agree) * 100))

if agree < 0.90:
    print("  WARNING: ordering is not consistent with periods_in_state.")
    print("  Inspect the dump below before reporting any day-based policy result.")
    sample_run = df_full[df_full["cath_run"] == 1].groupby(STAY_KEY, sort=False)
    for i, (key, g) in enumerate(sample_run):
        if len(g) >= 6:
            print("\n  stay", key)
            print(g[["episode_index", "periods_in_state", "interval_hours",
                     "cath_in_observed", "removed_in_period",
                     "reinsertion_in_period", "at_risk_cauti", "policy_day"]]
                  .head(14).to_string(index=False))
        if i >= 1:
            break


# --- 5. Measure the panel's existing post-removal window ---------------------
# Rather than assuming 48h, work out how long the panel itself keeps a patient
# CAUTI-at-risk after removal, then reuse that in the counterfactual.
df_full["_ih"] = df_full["interval_hours"].fillna(0.0).astype(float)
df_full["_t_end"] = df_full.groupby(STAY_KEY, sort=False)["_ih"].cumsum()
df_full["_t_start"] = df_full["_t_end"] - df_full["_ih"]
df_full["_t_rem"] = df_full["_t_end"].where(df_full["removed_in_period"] == 1)
df_full["_t_rem"] = df_full.groupby(STAY_KEY, sort=False)["_t_rem"].ffill()
df_full["hours_since_removal"] = df_full["_t_start"] - df_full["_t_rem"]

out_at_risk = (df_full["cath_out_observed"] == 1) & (df_full["at_risk_cauti"] == 1)
out_not_risk = (df_full["cath_out_observed"] == 1) & (df_full["at_risk_cauti"] == 0)

print("\n" + "-" * 70)
print("PANEL'S EXISTING POST-REMOVAL CAUTI WINDOW")
print("-" * 70)
print("Catheter-out rows still CAUTI-at-risk:", int(out_at_risk.sum()))
print("Catheter-out rows no longer at risk:  ", int(out_not_risk.sum()))

hrs_at_risk = df_full.loc[out_at_risk, "hours_since_removal"].dropna()
hrs_dropped = df_full.loc[out_not_risk, "hours_since_removal"].dropna()

# hours_since_removal is measured at the START of a period, so the window's
# upper edge is that plus the period's own duration.
end_at_risk = (
    df_full.loc[out_at_risk, "hours_since_removal"]
    + df_full.loc[out_at_risk, "interval_hours"].fillna(0.0)
).dropna()

measured = np.nan

if len(hrs_at_risk) > 0:
    print("\nHours since removal at the START of at-risk out-rows:")
    print(hrs_at_risk.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())
    w_hi = float(end_at_risk.quantile(0.99))
    print("\nUpper edge of the last at-risk period (99th pct): %.1f h" % w_hi)
else:
    w_hi = np.nan

if len(hrs_dropped) > 0:
    # The boundary: the earliest point at which the panel stops counting a
    # patient as at risk. This is the cleanest estimator of the window length.
    w_lo = float(hrs_dropped.quantile(0.05))
    print("Earliest point the panel drops a patient from the risk set (5th pct): %.1f h" % w_lo)
    measured = w_lo
elif np.isfinite(w_hi):
    measured = w_hi

if np.isfinite(w_hi) and np.isfinite(measured) and abs(w_hi - measured) > 24.0:
    print("  NOTE: the two estimates disagree by more than a day. The panel's window may")
    print("  not be a fixed number of hours - check before quoting a figure in the methods.")

# CAUTI events actually recorded after removal. If this is zero, the outcome is
# only ever recorded while the catheter is in, and the post-removal window is an
# assumption about exposure rather than something estimated from data.
if int(out_at_risk.sum()) > 0:
    print("\nCAUTI rate, catheterised rows:      %.5f"
          % float(df_full.loc[df_full["cath_in_observed"] == 1, TARGET].mean()))
    print("CAUTI rate, post-removal at-risk:   %.5f"
          % float(df_full.loc[out_at_risk, TARGET].mean()))
    print("CAUTI events post-removal:          %d"
          % int(df_full.loc[out_at_risk, TARGET].fillna(0).sum()))

# Use the measured window where it is sensible, otherwise the default
if np.isfinite(measured) and 6.0 <= measured <= 240.0:
    POST_REMOVAL_WINDOW_HOURS = float(np.round(measured))
    print("\nUsing MEASURED window: %.1f h" % POST_REMOVAL_WINDOW_HOURS)
else:
    POST_REMOVAL_WINDOW_HOURS = DEFAULT_WINDOW_HOURS
    print("\nUsing DEFAULT window: %.1f h (measurement unavailable or implausible)"
          % POST_REMOVAL_WINDOW_HOURS)

df_full = df_full.drop(columns=["_ih", "_t_end", "_t_start", "_t_rem"])


# --- 6. Run-length distribution and positivity -------------------------------
run_len = (
    df_full[df_full["cath_in_observed"] == 1]
    .groupby(RUN_KEY, sort=False)["policy_day"].max()
)
print("\n" + "-" * 70)
print("POSITIVITY - CATHETERISED PERIODS PER RUN")
print("-" * 70)
print("median %.0f  mean %.2f  max %.0f"
      % (run_len.median(), run_len.mean(), run_len.max()))
for day in range(1, 8):
    share = float((run_len >= day).mean()) * 100
    print("  reach day %d: %5.1f %%  -> expected violation %5.1f %%" % (day, share, 100 - share))

if DRY_RUN:
    print("\nDRY_RUN = True - stopping before model training.")
    raise SystemExit(0)


# Load data
print("Loading data ...")
df_full = pd.read_csv("dataset.csv", usecols=(all_columns_needed))
print("Data loaded. Shape:", df_full.shape)

missing_cols = [c for c in all_columns_needed if c not in df_full.columns]
if missing_cols:
    raise ValueError("These columns are missing from the dataset: " + str(missing_cols))
print("All expected columns are present.")

df_full["cath_run"] = run_start.groupby([df_full[c] for c in STAY_KEY], sort=False).cumsum()

# Sort into time order 
df_full = df_full.sort_values(
    ["subject_id", "hadm_id", "stay_id", "episode_index"]
).reset_index(drop=True)
print("Data sorted by stay then episode_index.")



# Training rows: the panel's own at-risk definition.
df_model = df_full[df_full["at_risk_cauti"] == 1].copy()

n_before = len(df_model)
df_model = df_model[df_model[TARGET].notna()].copy()
print("\nRows dropped for missing outcome:", n_before - len(df_model))
print("Rows used for model training:", len(df_model))
print("Rows kept for g-formula simulation:", len(df_full))

# Split using predefined "Split column"
df_train = df_model[df_model["split"] == "train"].copy()
df_test = df_model[df_model["split"] == "test"].copy()
print("Training rows:", len(df_train))
print("Test rows:    ", len(df_test))

X_train = df_train[PREDICTORS]
y_train = df_train[TARGET]
X_test = df_test[PREDICTORS]
y_test = df_test[TARGET]

# Group by column for k fold validation
groups = df_train["subject_id"]

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)

#Get Frequency of CAUTI in training and test data 
print("\nCAUTI frequency, TRAINING data:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(4))
print("\nCAUTI frequency, TEST data:")
print(y_test.value_counts())

# Get actual rate of CAUTI in dataset
print(y_test.value_counts(normalize=True).round(4))
print("\nbaseline rate:", round(y_test.mean(), 4))

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=PREDICTORS)
X_test = pd.DataFrame(imputer.transform(X_test), columns=PREDICTORS)
print("missing after:", "Training =", X_train.isna().sum().sum(),
      "Test =", X_test.isna().sum().sum())

# Save for use later
joblib.dump(imputer, "imputer_cauti.pkl")
print("Imputer saved to: imputer_cauti.pkl")

# Use Gradient Boosting Classifier ML model 
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    subsample=0.8,
    random_state=42
)

pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
group_kfold = GroupKFold(n_splits=5)

# Cross Validate with all three metrics
cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=group_kfold, groups=groups,
    scoring=["roc_auc", "average_precision", "neg_brier_score"]
)

print("\nCross-Validation Results (average across 5 folds)")
print("ROC-AUC:", round(cv_results["test_roc_auc"].mean(), 4))
print("AUPRC:  ", round(cv_results["test_average_precision"].mean(), 4))
print("Brier:  ", round((-cv_results["test_neg_brier_score"]).mean(), 4))

# Calibrate model using isotonic regression
calibrated_model = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)
print("Model trained and calibrated.")

#Save calibrated model results for use later
joblib.dump(calibrated_model, "model_cauti.pkl")
print("Calibrated model saved to: model_cauti.pkl")

y_prob = calibrated_model.predict_proba(X_test)[:, 1]
print("\nPredicted CAUTI probability summary (test set):")
print("Minimum: ", round(y_prob.min(), 4))
print("Maximum: ", round(y_prob.max(), 4))
print("Average: ", round(y_prob.mean(), 4))

# Get MACE scores for overall model calibration
fraction_pos_u, mean_pred_u = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
mace_uniform = np.mean(np.abs(fraction_pos_u - mean_pred_u))
fraction_pos_q, mean_pred_q = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
mace_quantile = np.mean(np.abs(fraction_pos_q - mean_pred_q))

print("\nCalibration check:")
print("Uniform-binned MACE: ", round(mace_uniform, 4), " (inflated by sparse high-probability bins)")
print("Quantile-binned MACE:", round(mace_quantile, 4), " (the meaningful figure for a rare outcome)")

# Get ROC-AUC (Discrimination), AUPRC (Precision), and Brier Scores (Accuracy)
roc_auc = roc_auc_score(y_test, y_prob)
auprc = average_precision_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n" + "=" * 55)
print("PROBABILITY MODEL EVALUATION")
print("=" * 55)
print("ROC-AUC:", round(roc_auc, 4))
print("AUPRC:  ", round(auprc, 4))
print("Brier:  ", round(brier, 4))

train_prob = calibrated_model.predict_proba(X_train)[:, 1]
print("Train ROC-AUC:", round(roc_auc_score(y_train, train_prob), 4))
print("Test ROC-AUC: ", round(roc_auc, 4))


# ==============================================================================
# PART 2 - G-FORMULA DATASET AND COUNTERFACTUAL TRAJECTORIES
# ==============================================================================

df_sim = df_full[df_full["split"] == "test"].copy()
df_sim = df_sim.sort_values(
    ["subject_id", "hadm_id", "stay_id", "episode_index"]
).reset_index(drop=True)

print("\nSimulation dataset created.")
print("Rows:    ", len(df_sim))
print("Patients:", df_sim["subject_id"].nunique())



# --- Stability criteria: coalesce arterial -> non-invasive -------------------
# Arterial O2 saturation is present on 13.7% of rows and arterial systolic on
# 42.8%, because both need an arterial line. Pulse oximetry (94.4%) and NBP
# (70.3%) are recorded on nearly everyone. Taking arterial where available and
# falling back to non-invasive raises assessability from 32.5% to 92.1%.
STABILITY_SOURCES = {
    "temp":  ["itemid_223762__last"],
    "hr":    ["itemid_220045__last"],
    "sbp":   ["itemid_220050__last", "itemid_220179__last"],
    "o2sat": ["itemid_220227__last", "itemid_220277__last"],
}

STABILITY_THRESHOLDS = {
    "temp_max": 38.0,
    "hr_max": 100.0,
    "sbp_min": 100.0,
    "o2sat_min": 94.0,
}



# --- Policies ----------------------------------------------------------------
# Thresholds above the max predicted probability can never fire, so the 25% and
# 50% variants are removed rather than reported as degenerate.
POLICIES = {
    "Observed Practice": {"type": "natural"},
    "Remove day 1": {"type": "day", "day": 1},
    "Remove day 2": {"type": "day", "day": 2},
    "Remove day 3": {"type": "day", "day": 3},
    "Remove day 4": {"type": "day", "day": 4},
    "Remove day 5": {"type": "day", "day": 5},
    "Remove day 6": {"type": "day", "day": 6},
    "Clinical stability": {"type": "stability"},
    "Clinical stability (≥ Day 3)": {"type": "stability_min_day", "min_day": 3},
    "Hybrid (stability OR day 4 cap)": {"type": "hybrid", "day_cap": 4},
    "Hybrid (stability OR day 5 cap)": {"type": "hybrid", "day_cap": 5},
    "Hybrid (stability OR day 6 cap)": {"type": "hybrid", "day_cap": 6},
    "Remove when CAUTI risk ≥ 1%": {"type": "risk_threshold", "threshold": 0.01},
    "Remove when CAUTI risk ≥ 2%": {"type": "risk_threshold", "threshold": 0.02},
    "Remove when CAUTI risk ≥ 5%": {"type": "risk_threshold", "threshold": 0.05},
    "Remove when CAUTI risk ≥ 10%": {"type": "risk_threshold", "threshold": 0.10},
}

# Function to see if policy removal trigger been met 
def check_removal_trigger(row, policy):

    ptype = policy["type"]

    # Fixed-day removal - policy_day counts catheterised periods since this
    # insertion, so it is unaffected by state changes elsewhere in the stay.
    if ptype == "day":
        return row["policy_day"] >= policy["day"]

    if ptype == "risk_threshold":
        return row["baseline_probability"] >= policy["threshold"]

    # Stability policies - coalesced vitals
    temp = row[STABILITY_COLS_USED["temp"]]
    hr = row[STABILITY_COLS_USED["hr"]]
    sbp = row[STABILITY_COLS_USED["sbp"]]
    o2sat = row[STABILITY_COLS_USED["o2sat"]]

    # missing vitals must never count as stable
    vitals_present = (
        pd.notna(temp) and pd.notna(hr)
        and pd.notna(sbp) and pd.notna(o2sat)
    )

    is_stable = (
        vitals_present
        and temp < STABILITY_THRESHOLDS["temp_max"]
        and hr < STABILITY_THRESHOLDS["hr_max"]
        and sbp > STABILITY_THRESHOLDS["sbp_min"]
        and o2sat > STABILITY_THRESHOLDS["o2sat_min"]
    )

    if ptype == "stability":
        return is_stable

    if ptype == "stability_min_day":
        return is_stable and (row["policy_day"] >= policy["min_day"])

    if ptype == "hybrid":
        return is_stable or (row["policy_day"] >= policy["day_cap"])

    return False

# Function to apply one policy to one catheter run-in
def create_counterfactual_run(run_rows, policy):

    run_cf = run_rows.copy()
    run_cf["positivity_violation"] = 0

    if policy["type"] == "natural":
        return run_cf

    run_cf["removed_in_period"] = 0

    # removal can only be decided on rows where the catheter was in place
    in_idx = run_cf.index[run_cf["cath_in_observed"] == 1]
    if len(in_idx) == 0:
        return run_cf

    removed = False

    for idx in in_idx:

        if removed:
            continue

        if check_removal_trigger(run_cf.loc[idx], policy):
            run_cf.loc[idx, "removed_in_period"] = 1
            removed = True

    # The run ended before the policy could fire. Removal defaults to the
    # observed end of catheterisation
    if not removed:
        run_cf.loc[in_idx[-1], "removed_in_period"] = 1
        run_cf["positivity_violation"] = 1

    return run_cf


def update_catheter_state(run_cf, window_hours):

    removed = run_cf["removed_in_period"].to_numpy()
    ih = run_cf["interval_hours"].fillna(0.0).to_numpy(dtype=float)

    # 1. catheter in place until the period after removal
    catheter_in = np.concatenate(([True], np.cumsum(removed[:-1]) == 0))

    # 1b. no removal recorded at all - do not let risk run past the observed
    # end of catheterisation
    if removed.sum() == 0:
        catheter_in = catheter_in & (run_cf["cath_in_observed"].to_numpy() == 1)

    run_cf["catheter_state_cf"] = np.where(catheter_in, "IN", "OUT")

    # 2. elapsed clock across this run
    t_end = np.cumsum(ih)
    t_start = t_end - ih

    # 3. time the simulated removal completed, carried forward
    t_removed = pd.Series(np.where(removed == 1, t_end, np.nan)).ffill().to_numpy()

    # 4. hours from removal to the start of each later period
    hours_since = t_start - t_removed

    # 5. out of catheter but still inside the post-removal window
    in_window = (~catheter_in) & ~np.isnan(hours_since) & (hours_since < window_hours)

    run_cf["at_risk_cauti_cf"] = (catheter_in | in_window).astype(int)

    return run_cf

# Zero the risk only once outside the post-removal window.
def predict_counterfactual(run_cf):


    probs = run_cf["baseline_probability"].to_numpy().copy()
    probs[run_cf["at_risk_cauti_cf"].to_numpy() == 0] = 0.0
    run_cf["predicted_probability"] = probs

    return run_cf


# ==============================================================================
# PART 3 - IMPLEMENT THE G FORMULA AND RESULTS
# ==============================================================================

def cumulative_risk(cf):
    
    p = cf["predicted_probability"].values
    
    return 1 - np.prod(1 - p)

patients_runs = []
 
for sid, patient_df in df_sim.groupby("subject_id", sort=False):
 
    runs = [
        run_df
        for run_key, run_df in patient_df.groupby(["hadm_id", "stay_id", "cath_run"], sort=False)
        if run_key[2] >= 1
    ]
 
    if runs:
        patients_runs.append(runs)

print("\nPatients with at least one catheter run:", len(patients_runs))
print("Catheter runs simulated per policy:      ", sum(len(r) for r in patients_runs))

results = []

for policy_name, policy in POLICIES.items():

    patient_risks = []
    at_risk_periods = []
    run_violations = []
    natural_agreement = []

    for runs in patients_runs:

        cf_parts = []

        # each run gets its own independent removal decision
        for run_df in runs:

            cf = create_counterfactual_run(run_df, policy)
            cf = update_catheter_state(cf, POST_REMOVAL_WINDOW_HOURS)
            cf = predict_counterfactual(cf)

            cf_parts.append(cf)
            run_violations.append(int(cf["positivity_violation"].iloc[0]))

            # Validation: under no intervention the reconstructed at-risk flag
            # should reproduce the panel's own at_risk_cauti.
            if policy["type"] == "natural":
                natural_agreement.append(
                    float((cf["at_risk_cauti_cf"] == cf["at_risk_cauti"]).mean())
                )

        cf_patient = pd.concat(cf_parts)
        patient_risks.append(cumulative_risk(cf_patient))
        at_risk_periods.append(int(cf_patient["at_risk_cauti_cf"].sum()))

    results.append({
        "Policy": policy_name,
        "Mean Risk": np.mean(patient_risks),
    })


policy_results = pd.DataFrame(results)
policy_results["Mean Risk (%)"] = (policy_results["Mean Risk"] * 100).round(3)

print("\n" + "=" * 90)
print("G-FORMULA CAUTI POLICY RISK ESTIMATES (cumulative per-patient risk)")

print("=" * 90)
print(policy_results[["Policy", "Mean Risk (%)"]].to_string(index=False))
