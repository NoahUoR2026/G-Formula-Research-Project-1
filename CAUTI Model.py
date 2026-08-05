# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:17:48 2026

@author: noaha
"""

'''
CAUTI PROBABILITY MODEL + G-FORMULA SIMULATION
'''

# Import Libraries
import pandas as pd                                      
import numpy as np                                       
import joblib 
import matplotlib.pyplot as plt                                          

from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting Model
from sklearn.impute import SimpleImputer                 
from sklearn.model_selection import GroupKFold, cross_validate  # for cross-validation
from sklearn.pipeline import Pipeline                    # for chaining steps together
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # evaluation metrics
from sklearn.calibration import calibration_curve, CalibratedClassifierCV  # for calibration


# ==============================================================================
# PART 1 - BUILD THE CAUTI PROBABILITY MODEL
# ==============================================================================


# Target patients at risk of CAUTI
TARGET = "cauti_in_period"

# Predictor list
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

# ID's, flags and outcomes needed alongside the predictors
EXTRA_COLS = [
    "subject_id",           # unique patient identifier
    "hadm_id",              # Hospital admission ID
    "stay_id",              # ICU stay ID
    "episode_index",        # which catheter episode this row belongs to
    "periods_in_state",     # time periods in current state (used for sorting and for policies)
    "interval_hours",       # hours since last observation
    "removed_in_period",    # was the catheter removed in this time period? (the intervention)
    "at_risk_cauti",        # is this patient currently at risk of CAUTI?
    "at_risk_reinsertion",  # is this patient currently at risk of reinsertion?
    "cauti_in_period",      # did the patient get CAUTI in this period? (outcome)
    "reinsertion_in_period",# was the catheter reinserted in this period? (second outcome)
    "split",                # tells us if this row is training or test data
    "itemid_220179__last",  # NBP Systolic for stability criteria       
    "itemid_220277__last",  # O2 Sat, pulse for stability criteria   
]

# Combine predictor columns and extra columns into one list 
all_columns_needed = list(set(PREDICTORS + EXTRA_COLS))

# Load the data 
print("Loading data...")
df_full = pd.read_csv("Dataset.csv", usecols=all_columns_needed)
print("Data loaded. Shape:", df_full.shape)  # prints (rows, columns)


# Check if any expected columns are missing from the loaded data
missing_cols = []

for col in all_columns_needed:

    if col not in df_full.columns:
        missing_cols.append(col)

# If any are missing, stop the script and output which ones
if len(missing_cols) > 0:
    raise ValueError("These columns are missing from the dataset:", missing_cols)

print("All expected columns are present.")

# catheter state
df_full["catheter_in"] = (df_full["at_risk_reinsertion"] == 0).astype(int)

# add it to the predictor list so it is used in training AND every prediction
PREDICTORS.append("catheter_in")
print("Added 'catheter_in' feature. In-catheter rows:",
      int(df_full["catheter_in"].sum()),
      "| post-removal window rows:",
      int(((df_full["catheter_in"] == 0) & (df_full["at_risk_cauti"] == 1)).sum()))


# Get catheter episode ID
STAY_KEY = ["subject_id", "hadm_id", "stay_id"]

# sort into time order first - episode_index IS a valid within-stay row counter,
# which is exactly what we need as a clock
df_full = df_full.sort_values(STAY_KEY + ["episode_index"]).reset_index(drop=True)

_in = df_full["catheter_in"] == 1
_prev_in = _in.groupby([df_full[c] for c in STAY_KEY], sort=False).shift(1).fillna(False).astype(bool)
_start = (_in & ~_prev_in).astype(int)
df_full["cath_episode"] = _start.groupby([df_full[c] for c in STAY_KEY], sort=False).cumsum()

# keep the observed state under its own name - update_catheter_state overwrites
# catheter_in with the COUNTERFACTUAL state later
df_full["cath_in_observed"] = df_full["catheter_in"]

# Key that uniquely identifies one catheter episode
EPISODE_KEY = STAY_KEY + ["cath_episode"]

# drop rows before the first insertion - they belong to no episode
df_full = df_full[df_full["cath_episode"] >= 1].reset_index(drop=True)

print("Data sorted into time order per episode.")
print("Catheter episodes detected:", df_full.groupby(EPISODE_KEY).ngroups)
_rpe = df_full.groupby(EPISODE_KEY).size()
print("Rows per episode: mean %.2f  median %.0f  max %d"
      % (_rpe.mean(), _rpe.median(), _rpe.max()))
if _rpe.mean() < 1.5:
    raise ValueError(
        "Episodes average under 1.5 rows - the episode key is still fragmenting "
        "the panel and every policy will collapse to the same risk."
    )


# filter to at risk rows only for model training 
df_model = df_full[df_full["at_risk_cauti"] == 1].copy()  # .copy() prevents accidental edits to df_full

print("Rows used for model training (at risk only):", len(df_model))
print("Rows kept aside for g-formula simulation (all rows):", len(df_full))


# split into train and test data using dataset 'split' column
df_train = df_model[df_model["split"] == "train"].copy()  # rows for training the model
df_test  = df_model[df_model["split"] == "test"].copy()   # rows for evaluating the model

print("Training rows:", len(df_train))
print("Test rows:",     len(df_test))


# X = the input features the model uses to make predictions
# y = the thing being predicted (did they get CAUTI?)

X_train = df_train[PREDICTORS]   # training features
y_train = df_train[TARGET]       # training outcome

X_test = df_test[PREDICTORS]     # test features
y_test = df_test[TARGET]         # test outcome

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Get patient IDs from training data for grouped cross-validation.
groups = df_train["subject_id"]


# How common is CAUTI in actual data
print("\nHow often does CAUTI occur in the TRAINING data?")
print(y_train.value_counts())                          # raw counts
print(y_train.value_counts(normalize=True).round(4))   # as proportions

print("\nHow often does CAUTI occur in the TEST data?")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True).round(4))

# The baseline rate is the proportion of positive cases.
# A model that just guesses randomly would score around this value on AUPRC.
baseline_rate = y_test.mean()
print("\nbaseline rate:", round(baseline_rate, 4))

# Check missingness
missing = (
    df_full[PREDICTORS]
    .isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

missing_table = pd.DataFrame({
    "Missing_Percentage": missing
})

print("\n" + "=" * 55)
print("MISSING DATA SUMMARY (MODEL PREDICTORS)")
print("=" * 55)
print(missing_table)

plt.figure(figsize=(14, 8))
missing.plot(kind="bar")

plt.title("Missing Data Percentage for Model Predictor Variables")
plt.xlabel("Predictor")
plt.ylabel("% Missing")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Median imputation, fit on train only
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=PREDICTORS)
X_test = pd.DataFrame(imputer.transform(X_test), columns=PREDICTORS)

# fill in missing values
imputer = SimpleImputer(strategy="median")             # create the imputer
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=PREDICTORS)  # fit on training data and fill missing values
X_test = pd.DataFrame(imputer.transform(X_test), columns=PREDICTORS)        # apply same medians to test data (no refitting)

print("missing after:", "Training =", X_train.isna().sum().sum(), "Test =", X_test.isna().sum().sum())

# Save the fitted imputer to disk to use later
joblib.dump(imputer, "imputer_cauti.pkl")
print("Imputer saved to: imputer_cauti.pkl")


# Set up the model 
model = GradientBoostingClassifier(
    n_estimators=300,    # build 300 decision trees
    learning_rate=0.03,  # each tree contributes a small amount (helps avoid overfitting)
    max_depth=2,         # each tree is kept shallow (simple)
    subsample=0.8,       # each tree only sees 80% of the training data (adds variety)
    random_state=42      # makes results reproducible
)

# chain the imputer and model together into one object.
pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

# Ensure the same patient never appears in both the training fold and the validation fold during cross-validation.
group_kfold = GroupKFold(n_splits=5)  # split into 5 folds

# Run cross-validation and collect three metrics
cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=group_kfold,
    groups=groups,
    scoring=["roc_auc", "average_precision", "neg_brier_score"]
)

# Print average scores across the 5 folds
print("Cross-Validation Results (average across 5 folds)")
# how well the model ranks patients
print("ROC-AUC:", round(cv_results["test_roc_auc"].mean(), 4))
# precision-recall performance            
print("AUPRC:  ", round(cv_results["test_average_precision"].mean(), 4)) 
# overall probability accuracy 
print("Brier:  ", round((-cv_results["test_neg_brier_score"]).mean(), 4)) 



# train the final model and calibrate
calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,  # the model pipeline to calibrate
    method="isotonic",   # isotonic regression calibration (best for rare outcomes)
    cv=5                 # uses 5-fold cross-fitting internally (avoids circularity vs cv="prefit")
)

# Train the calibrated model on all training data
calibrated_model.fit(X_train, y_train)
print("Model trained and calibrated.")

# Save the calibrated model to disk for use later
joblib.dump(calibrated_model, "model_cauti.pkl")
print("Calibrated model saved to: model_cauti.pkl")



# Generate the predicted probabilities
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("\nPredicted CAUTI probability summary (test set):")
print("Minimum: ", round(y_prob.min(), 4))
print("Maximum: ", round(y_prob.max(), 4))
print("Average: ", round(y_prob.mean(), 4))



# Check how well calibrated the probability outputs are
# Calibration means: when the model says 10% probability, does CAUTI happen 10% of the time?
fraction_pos_uniform, mean_pred_uniform = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
mace_uniform = np.mean(np.abs(fraction_pos_uniform - mean_pred_uniform))

# calibration using quantile bins
fraction_pos_quantile, mean_pred_quantile = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
mace_quantile = np.mean(np.abs(fraction_pos_quantile - mean_pred_quantile))

print("\nCalibration check:")
print("Uniform-binned MACE: ", round(mace_uniform, 4), " (can be inflated by sparse high-probability bins)")
print("Quantile-binned MACE:", round(mace_quantile, 4), " (more reliable diagnostic for a rare outcome)")



# Evalution metrics 

# how well the model separates cases from non-cases
roc_auc = roc_auc_score(y_test, y_prob)
# performance under class imbalance           
auprc   = average_precision_score(y_test, y_prob) 
# overall error in predicted probabilities
brier   = brier_score_loss(y_test, y_prob)        

print("\n" + "=" * 55)
print("PROBABILITY MODEL EVALUATION")
print("=" * 55)
print("ROC-AUC:", round(roc_auc, 4))
print("AUPRC:  ", round(auprc, 4))
print("Brier:  ", round(brier, 4))

# Check for overfitting: if train ROC-AUC is much higher than test, the model has overfit
train_prob = calibrated_model.predict_proba(X_train)[:, 1]
print("Train ROC-AUC:", round(roc_auc_score(y_train, train_prob), 4))
print("Test ROC-AUC: ", round(roc_auc, 4))
print("(A big gap between these two suggests overfitting)")


# ==============================================================================
# PART 2 - BUILD G FORMULA DATASET AND COUNTERFACTUAL TRAJECTORIES 
# ==============================================================================

# How many hours after removal CAUTI risk should remain active for.
POST_REMOVAL_RISK_WINDOW_HOURS = 48

# Build the simulation dataset for the g formula to be implemented
df_sim = df_full[df_full["split"] == "test"].copy()

# Make sure it is still in time order (within each catheter episode)
df_sim = df_sim.sort_values(
    STAY_KEY + ["episode_index"]
).reset_index(drop=True)

print("\nSimulation dataset created.")
print("Rows:    ", len(df_sim))
print("Episodes:", df_sim.groupby(EPISODE_KEY).ngroups)

# Now restrict down to just the rows where the patient was actually at risk of CAUTI.
df_sim_atrisk = df_sim[df_sim["at_risk_cauti"] == 1].copy()
print("At-risk rows available for policy simulation:", len(df_sim_atrisk))

# Period number
df_sim["period_number"] = df_sim.groupby(EPISODE_KEY).cumcount() + 1

# If still catheterised probabilities
print("Precomputing 'if still catheterised' CAUTI probabilities...")

X_if_in = df_sim[PREDICTORS].copy()
X_if_in["periods_in_state"] = df_sim["period_number"]
X_if_in["catheter_in"] = 1     # force the in-catheter state

df_sim["predicted_probability_if_in"] = calibrated_model.predict_proba(X_if_in)[:, 1]

print("Done. Mean 'if in' risk per row:", round(df_sim["predicted_probability_if_in"].mean(), 5))

# Sanity check
X_if_out = X_if_in.copy()
X_if_out["catheter_in"] = 0
_mean_if_out = calibrated_model.predict_proba(X_if_out)[:, 1].mean()
print("Mean 'if in' risk :", round(df_sim['predicted_probability_if_in'].mean(), 5))
print("Mean 'if window' risk:", round(float(_mean_if_out), 5),
      "  <-- should be clearly lower than 'if in'")

# Predictor list used for scoring rows in the post-removal 48h window
PREDICTORS_CF = [
    "periods_in_state_cf" if col == "periods_in_state" else col
    for col in PREDICTORS
]


# Clinical stability criteria 
STABILITY_SOURCES = {
    "temp":  ["itemid_223762__mean"],
    "hr":    ["itemid_220045__mean"],
    "sbp":   ["itemid_220050__mean", "itemid_220179__last"],
    "o2sat": ["itemid_220227__last", "itemid_220277__last"],
}

STABILITY_COLS = {}

for _key, _sources in STABILITY_SOURCES.items():
    _s = df_sim.groupby(STAY_KEY, sort=False)[_sources[0]].ffill()
    for _alt in _sources[1:]:
        _s = _s.fillna(df_sim.groupby(STAY_KEY, sort=False)[_alt].ffill())
    df_sim["stab_" + _key] = _s
    STABILITY_COLS[_key] = "stab_" + _key

_vitals_ok = df_sim[list(STABILITY_COLS.values())].notna().all(axis=1)
print("Rows with all four stability vitals available: %.1f %%"
      % (float(_vitals_ok.mean()) * 100))

STABILITY_THRESHOLDS = {
    "temp_max": 38.0,
    "hr_max": 100.0,
    "sbp_min": 100.0,
    "o2sat_min": 94.0,
}


# Define the counterfactual removal policies.
POLICIES = {
    # Natural practice
    "Observed Practice": {
        "type": "natural"
    },
    # Fixed removal days
    "Remove day 1": {"type": "day", "day": 1},
    "Remove day 2": {"type": "day", "day": 2},
    "Remove day 3": {"type": "day", "day": 3},
    "Remove day 4": {"type": "day", "day": 4},
    "Remove day 5": {"type": "day", "day": 5},
    "Remove day 6": {"type": "day", "day": 6},
    # Remove as soon as clinically stable
    "Clinical stability": {"type": "stability"},
    # Stable only after minimum of 3 days
    "Clinical stability (≥ Day 3)": {"type": "stability_min_day", "min_day": 3},
    # Hybrid policies
    "Hybrid (stability OR day 4 cap)": {"type": "hybrid", "day_cap": 4},
    "Hybrid (stability OR day 5 cap)": {"type": "hybrid", "day_cap": 5},
    "Hybrid (stability OR day 6 cap)": {"type": "hybrid", "day_cap": 6},
    # CAUTI probability-triggered policies
    "Remove when CAUTI risk ≥ 1%":  {"type": "risk_threshold", "threshold": 0.01},
    "Remove when CAUTI risk ≥ 2%":  {"type": "risk_threshold", "threshold": 0.02},
    "Remove when CAUTI risk ≥ 5%":  {"type": "risk_threshold", "threshold": 0.05},
    "Remove when CAUTI risk ≥ 10%": {"type": "risk_threshold", "threshold": 0.10},
    "Remove when CAUTI risk ≥ 25%": {"type": "risk_threshold", "threshold": 0.25},
    "Remove when CAUTI risk ≥ 50%": {"type": "risk_threshold", "threshold": 0.50},
}


# Function to check whether a policy's removal trigger is met on this row.
def check_removal_trigger(row, policy):

    ptype = policy["type"]

    # Fixed-day removal to trigger once period_number reaches the target day
    if ptype == "day":
        return row["period_number"] >= policy["day"]

    # Risk-threshold removal 
    if ptype == "risk_threshold":
        return row["predicted_probability_if_in"] >= policy["threshold"]

    # Everything else (stability-based policies) needs the four vitals
    temp = row[STABILITY_COLS["temp"]]
    hr = row[STABILITY_COLS["hr"]]
    sbp = row[STABILITY_COLS["sbp"]]
    o2sat = row[STABILITY_COLS["o2sat"]]

    # only count as stable if all four vitals are actually recorded 
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
        return is_stable and (row["period_number"] >= policy["min_day"])

    if ptype == "hybrid":
        return is_stable or (row["period_number"] >= policy["day_cap"])

    # Shouldn't get here if POLICIES is defined correctly
    return False


# Function to create a counterfactual episode
def create_counterfactual_patient(patient_rows, policy):

    # create a copy of the episode
    patient_cf = patient_rows.copy()

    ptype = policy["type"]

    # Natural practice 
    if ptype == "natural":
        patient_cf["positivity_violation"] = 0
        return patient_cf

    # reset removal flag - this is a counterfactual so we decide removal fresh
    patient_cf["removed_in_period"] = 0
    patient_cf["positivity_violation"] = 0

    # a removal can only be decided on periods where the catheter was really in
    in_idx = patient_cf.index[patient_cf["cath_in_observed"] == 1]

    if len(in_idx) == 0:
        return patient_cf

    removed = False

    # loop through the catheterised periods of the episode
    for idx in in_idx:

        # once removal has already been triggered, no further removals
        if removed:
            continue

        row = patient_cf.loc[idx]

        # check whether THIS policy's trigger condition is met yet
        trigger = check_removal_trigger(row, policy)

        if trigger:
            patient_cf.loc[idx, "removed_in_period"] = 1
            removed = True

    # The episode ended before the policy 
    if not removed:
        # removal to the last catheterised period.
        obs_rem = patient_rows.index[patient_rows["removed_in_period"] == 1]
        if len(obs_rem) > 0:
            patient_cf.loc[obs_rem[0], "removed_in_period"] = 1
        else:
            patient_cf.loc[in_idx[-1], "removed_in_period"] = 1
        patient_cf["positivity_violation"] = 1

    return patient_cf


# Function to update catheter state, the 48h CAUTI at-risk window
def update_catheter_state(patient_cf,
                          post_removal_window_hours=POST_REMOVAL_RISK_WINDOW_HOURS):

    patient_cf = patient_cf.copy()

    removed = patient_cf["removed_in_period"].to_numpy()
    hours = patient_cf["interval_hours"].to_numpy()
    period_number = patient_cf["period_number"].to_numpy()

    n = len(patient_cf)

    # catheter in or out
    catheter_in = np.ones(n, dtype=bool)

    removal_index_arr = np.where(removed == 1)[0]

    if len(removal_index_arr) > 0:
        removal_index = removal_index_arr[0]
        # the removal row itself still counts as IN 
        catheter_in[removal_index + 1:] = False
    else:
        removal_index = None

    patient_cf["catheter_state_cf"] = np.where(catheter_in, "IN", "OUT")

    # expose the counterfactual state as the model feature too
    patient_cf["catheter_in"] = catheter_in.astype(int)


    # Hours since removal 
    hours_since_removal = np.zeros(n)

    if removal_index is not None:
        running_total = 0.0
        for i in range(removal_index + 1, n):
            running_total += hours[i]
            hours_since_removal[i] = running_total


    # CAUTI at-risk window: IN, or OUT but still within 48h of removal

    at_risk_cf = (
        catheter_in
        | ((~catheter_in) & (hours_since_removal <= post_removal_window_hours))
    )
    patient_cf["at_risk_cauti_cf"] = at_risk_cf.astype(int)

    #periods_in_state_cf 
    periods_in_state_cf = np.zeros(n, dtype=int)
    out_count = 0
    for i in range(n):
        if catheter_in[i]:
            periods_in_state_cf[i] = period_number[i]
        else:
            out_count += 1
            periods_in_state_cf[i] = out_count
    patient_cf["periods_in_state_cf"] = periods_in_state_cf

    return patient_cf


# ==============================================================================
# PART 3 - IMPLEMENT THE G FORMULA AND RESULTS
# ==============================================================================

# Function to run G formula to get cumulative risk of CAUTI
def cumulative_risk(patient_cf):
    p = patient_cf["predicted_probability"].values
    return 1 - np.prod(1 - p)


results = []

# evaluate each policy
for policy_name, policy in POLICIES.items():

    episode_frames = []

    # run simulation separately for each catheter episode =
    for _, episode in df_sim.groupby(EPISODE_KEY):

        cf = create_counterfactual_patient(episode, policy)
        cf = update_catheter_state(cf)
        episode_frames.append(cf)

    all_cf = pd.concat(episode_frames)
    all_cf["predicted_probability"] = 0.0

    # in rows reuse the precomputed "if still catheterised" probability =
    in_mask = all_cf["catheter_state_cf"] == "IN"
    all_cf.loc[in_mask, "predicted_probability"] = all_cf.loc[in_mask, "predicted_probability_if_in"]

    # out rows still inside the 48h window: 
    window_mask = (all_cf["catheter_state_cf"] == "OUT") & (all_cf["at_risk_cauti_cf"] == 1)
    window_rows = all_cf.loc[window_mask]

    if len(window_rows) > 0:
        X = window_rows[PREDICTORS_CF].copy()
        X.columns = PREDICTORS         # rename periods_in_state_cf to periods_in_state for the model
        X["catheter_in"] = 0           # within the 48h post-removal window
        X_imputed = pd.DataFrame(imputer.transform(X), columns=PREDICTORS, index=X.index)
        probs = calibrated_model.predict_proba(X_imputed)[:, 1]
        all_cf.loc[window_rows.index, "predicted_probability"] = probs

    # rows beyond the 48h window (OUT & at_risk_cauti_cf == 0) stay at 0

    # cumulative risk per episode, then average across all episodes
    episode_risks = []
    for _, group in all_cf.groupby(EPISODE_KEY):
        episode_risks.append(cumulative_risk(group))

    results.append({
        "Policy": policy_name,
        "Mean Risk": np.mean(episode_risks),
        "Positivity Viol (%)": round(
            float(all_cf.groupby(EPISODE_KEY)["positivity_violation"].max().mean()) * 100, 1),
    })

# convert results into dataframe for policy comparison
policy_results = pd.DataFrame(results)
policy_results["Mean Risk (%)"] = (policy_results["Mean Risk"] * 100).round(3)

# Get results
print("\n" + "=" * 70)
print("G-FORMULA CAUTI POLICY RISK ESTIMATES (cumulative per-episode risk)")
print("=" * 70)
print(policy_results.to_string(index=False))
