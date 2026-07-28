
'''
CAUTI PROBABILITY MODEL + G-FORMULA SIMULATION
'''

# Import Libraries
import pandas as pd                                      
import numpy as np                                       
import joblib                                           

from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting Model
from sklearn.impute import SimpleImputer                 
from sklearn.model_selection import GroupKFold, cross_validate  # for cross-validation
from sklearn.pipeline import Pipeline                    # for chaining steps together
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # evaluation metrics
from sklearn.calibration import calibration_curve, CalibratedClassifierCV  # for calibration


# ==============================================================================
# PART 1 - BUILD THE CAUTI PROBABILITY MODEL
# ==============================================================================


# Target Column for prediction (did the patient get a CAUTI in this period?)
TARGET = "cauti_in_period"

# These are the patient characteristics used to make the prediction.
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


# Sort the data into time order
df_full = df_full.sort_values(
    ["subject_id", "episode_index", "periods_in_state"]
).reset_index(drop=True)  # reset_index gives a clean 0,1,2 row index after sorting
print("Data sorted into time order per patient.")


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

# Build the simulation dataset for the g formula to be implemented
# The g-formula needs the FULL patient timeline (not just at-risk rows)
df_sim = df_full[df_full["split"] == "test"].copy()

# Make sure it is still in time order
df_sim = df_sim.sort_values(
    ["subject_id", "episode_index", "periods_in_state"]
).reset_index(drop=True)

print("\nSimulation dataset created.")
print("Rows:    ", len(df_sim))
print("Patients:", df_sim["subject_id"].nunique())

# Now restrict down to just the rows where the patient was actually at risk of CAUTI.
df_sim_atrisk = df_sim[df_sim["at_risk_cauti"] == 1].copy()
print("At-risk rows available for policy simulation:", len(df_sim_atrisk))

# ==============================================================================
# PRECOMPUTE BASELINE CAUTI PROBABILITIES
# ==============================================================================

print("Precomputing baseline CAUTI probabilities...")

df_sim["baseline_probability"] = calibrated_model.predict_proba(
    df_sim[PREDICTORS]
)[:, 1]

print("Done.")

# Key that uniquely identifies one catheter episode - simulation must run within this, independently
EPISODE_KEY = ["subject_id", "hadm_id", "stay_id", "episode_index"]

# Column to sort by so rows are in correct time order before any sequential logic
TIME_KEY = "interval_hours"


# Clinical stability criteria - four vitals need to all be within safe
# ranges before a patient counts as "stable" for the stability-based
# policies below. Using __last for all four so the rule is checking the
# most recent reading, not an average over the whole stay.
STABILITY_COLS = {
    "temp": "itemid_223762__mean",    # Temperature (C) < 38 C
    "hr": "itemid_220045__mean",      # Heart Rate < 100
    "sbp": "itemid_220050__mean",     # Arterial BP Systolic > 100
    "o2sat": "itemid_220227__last",   # Arterial O2 Saturation > 94%
}

STABILITY_THRESHOLDS = {
    "temp_max": 38.0,
    "hr_max": 100.0,
    "sbp_min": 100.0,
    "o2sat_min": 94.0,
}


# Define the counterfactual removal policies.
# Each policy is a dictionary describing HOW removal is triggered:
#   "natural"        - no intervention, use real observed practice
#   "day"            - force removal once a fixed day is reached
#   "stability"      - remove as soon as all 4 vitals meet the criteria
#   "stability_min_day" - stability AND a minimum number of days elapsed
#   "hybrid"         - remove at whichever comes first: stability OR a day cap
#   "risk_threshold" - remove as soon as predicted CAUTI risk crosses a
#                      given probability threshold
POLICIES = {
    # Natural practice
    "Observed Practice": {
        "type": "natural"
    },
    # Fixed removal days
    "Remove day 1": {
        "type": "day",
        "day": 1
    },
    "Remove day 2": {
        "type": "day",
        "day": 2
    },
    "Remove day 3": {
        "type": "day",
        "day": 3
    },
    "Remove day 4": {
        "type": "day",
        "day": 4
    },
    "Remove day 5": {
        "type": "day",
        "day": 5
    },
    "Remove day 6": {
        "type": "day",
        "day": 6
    },
    # Remove as soon as clinically stable
    "Clinical stability": {
        "type": "stability"
    },
    # Stable only after minimum of 3 days
    "Clinical stability (≥ Day 3)": {
        "type": "stability_min_day",
        "min_day": 3
    },
    # Hybrid policies
    "Hybrid (stability OR day 4 cap)": {
        "type": "hybrid",
        "day_cap": 4
    },
    "Hybrid (stability OR day 5 cap)": {
        "type": "hybrid",
        "day_cap": 5
    },
    "Hybrid (stability OR day 6 cap)": {
        "type": "hybrid",
        "day_cap": 6
    },
    # CAUTI probability-triggered policies
    "Remove when CAUTI risk ≥ 1%": {
        "type": "risk_threshold",
        "threshold": 0.01
    },
    "Remove when CAUTI risk ≥ 2%": {
        "type": "risk_threshold",
        "threshold": 0.02
    },
    "Remove when CAUTI risk ≥ 5%": {
        "type": "risk_threshold",
        "threshold": 0.05
    },
    "Remove when CAUTI risk ≥ 10%": {
        "type": "risk_threshold",
        "threshold": 0.10
    },
    "Remove when CAUTI risk ≥ 25%": {
        "type": "risk_threshold",
        "threshold": 0.25
    },
    "Remove when CAUTI risk ≥ 50%": {
        "type": "risk_threshold",
        "threshold": 0.50
    }
}


# Function to check whether a policy's removal trigger is met on this row.
# Kept separate from create_counterfactual_patient so each policy type's
# logic is easy to read in one place, rather than buried in the main loop.
def check_removal_trigger(row, policy):

    ptype = policy["type"]

    # Fixed-day removal - trigger once periods_in_state reaches the target day
    if ptype == "day":
        return row["periods_in_state"] >= policy["day"]

    # Risk-threshold removal - need the model's predicted CAUTI probability
    # for THIS row, using the row's real observed features, to check
    # whether it has crossed the threshold yet
    if ptype == "risk_threshold":

        return row["baseline_probability"] >= policy["threshold"]

    # Everything else (stability-based policies) needs the four vitals
    temp = row[STABILITY_COLS["temp"]]
    hr = row[STABILITY_COLS["hr"]]
    sbp = row[STABILITY_COLS["sbp"]]
    o2sat = row[STABILITY_COLS["o2sat"]]

    # only count as stable if all four vitals are actually recorded -
    # missing vitals should never count as "stable"
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
        return is_stable and (row["periods_in_state"] >= policy["min_day"])

    if ptype == "hybrid":
        return is_stable or (row["periods_in_state"] >= policy["day_cap"])

    # Shouldn't get here if POLICIES is defined correctly
    return False


# Function to create a counterfactual episode
def create_counterfactual_patient(patient_rows, policy):

    # create a copy of patients episode
    patient_cf = patient_rows.copy()

    ptype = policy["type"]

    # Natural practice - use the real observed data as-is, no intervention
    if ptype == "natural":
        return patient_cf

    # reset removal flag - don't trust whatever was in the real data,
    # this is a counterfactual so we decide removal fresh
    patient_cf["removed_in_period"] = 0

    removed = False

    # loop through each time period in patients catheter episode
    for idx in patient_cf.index:

        # once removal has already been triggered, no further removals
        if removed:
            continue

        row = patient_cf.loc[idx]

        # check whether THIS policy's trigger condition is met yet
        trigger = check_removal_trigger(row, policy)

        if trigger:
            # simulate catheter removal
            patient_cf.loc[idx, "removed_in_period"] = 1
            removed = True

    # return counterfactual patient timeline
    return patient_cf



# Function to update the catheter state
def update_catheter_state(patient_cf):

    removed = patient_cf["removed_in_period"].to_numpy()

    catheter_in = np.concatenate(
        ([True], np.cumsum(removed[:-1]) == 0)
    )

    patient_cf["catheter_state_cf"] = np.where(catheter_in, "IN", "OUT")

    return patient_cf


# Function to predict counterfactual CAUTI risk 
def predict_counterfactual(patient_cf):

    probs = patient_cf["baseline_probability"].to_numpy().copy()

    probs[patient_cf["catheter_state_cf"] == "OUT"] = 0.0

    patient_cf["predicted_probability"] = probs

    return patient_cf

# ==============================================================================
# PART 3 - IMPLEMENT THE G FORMULA AND RESULTS
# ==============================================================================

# Function to run G formula to get cumulative risk of CAUTI
def cumulative_risk(patient_cf):

    # extract predicted CAUTI for each time period
    p = patient_cf["predicted_probability"].values 

    # calculate probability of experiencing at least one CAUTI
    return 1 - np.prod(1 - p) 

# empty list to store results
# Cache patient groups once
patients = list(df_sim.groupby("subject_id"))

results = [] 

 # evaluate each policy
for policy_name, policy in POLICIES.items():

    # store cumulative risk
    patient_risks = [] 

    # run simulation separately for each patient
    for _, patient in patients: 

        # apply selected policy
        cf = create_counterfactual_patient(patient, policy)

        # update catheter status after removal
        cf = update_catheter_state(cf) 
        
        # predict CAUTI risk under hypothetical scenario
        cf = predict_counterfactual(cf) # predict CAUTI risk under hypothetical scenario

        # calculate overall CAUTI risk
        patient_risks.append(cumulative_risk(cf)) 

# calculate average CAUTI risk across all patients
    results.append({"Policy": policy_name,"Mean Risk": np.mean(patient_risks)})

# convert results into dataframe for policy comparison
policy_results = pd.DataFrame(results)
policy_results["Mean Risk (%)"] = (policy_results["Mean Risk"] * 100).round(3)

# Get results
print("\n" + "=" * 70)
print("G-FORMULA CAUTI POLICY RISK ESTIMATES (cumulative per-patient risk)")
print("=" * 70)
print(policy_results.to_string(index=False))