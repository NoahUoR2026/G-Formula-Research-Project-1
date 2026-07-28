"""
REINSERTION PROBABILITY MODEL + G-FORMULA SIMULATION
(Day + Stability + Hybrid + CAUTI-RISK-THRESHOLD policies)


"""

# Import Libraries
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
# PART 1 - BUILD THE REINSERTION PROBABILITY MODEL
# ==============================================================================

TARGET = "reinsertion_in_period"

PREDICTORS = [
    "age",
    "sex_M",
    "episode_index",
    "itemid_229371__last",
    "itemid_220739__last",
    "itemid_223901__last",
    "itemid_223900__last",
    "itemid_220045__mean",
    "itemid_220052__mean",
    "itemid_220210__mean",
    "itemid_223762__mean",
    "itemid_220615__last",
    "itemid_225624__last",
    "itemid_227471__last",
    "itemid_220546__last",
    "itemid_229357__last",
    "itemid_220228__last",
    "itemid_227457__last",
    "itemid_220645__last",
    "itemid_227442__last",
    "itemid_220602__last",
    "itemid_220621__last",
    "itemid_227456__last",
    "itemid_220644__last",
    "itemid_220587__last",
    "itemid_225690__last",
    "itemid_225612__last",
    "itemid_227073__last",
    "itemid_220235__last",
    "itemid_220224__last",
    "itemid_220227__last",
    "itemid_220051__mean",
    "itemid_220050__mean",
]

EXTRA_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "episode_index",
    "periods_in_state",
    "interval_hours",
    "removed_in_period",
    "at_risk_cauti",
    "at_risk_reinsertion",
    "cauti_in_period",
    "reinsertion_in_period",
    "split",
]

all_columns_needed = list(set(PREDICTORS + EXTRA_COLS))

print("Loading data...")
df_full = pd.read_csv("Dataset.csv", usecols=all_columns_needed)
print("Data loaded. Shape:", df_full.shape)

missing_cols = [c for c in all_columns_needed if c not in df_full.columns]
if len(missing_cols) > 0:
    raise ValueError("These columns are missing from the dataset:", missing_cols)
print("All expected columns are present.")

df_full = df_full.sort_values(
    ["subject_id", "episode_index", "periods_in_state"]
).reset_index(drop=True)
print("Data sorted into time order per patient.")

df_model = df_full[df_full["at_risk_reinsertion"] == 1].copy()
print("Rows used for model training (at risk only):", len(df_model))
print("Rows kept aside for g-formula simulation (all rows):", len(df_full))

df_train = df_model[df_model["split"] == "train"].copy()
df_test = df_model[df_model["split"] == "test"].copy()
print("Training rows:", len(df_train))
print("Test rows:", len(df_test))

X_train = df_train[PREDICTORS]
y_train = df_train[TARGET]
X_test = df_test[PREDICTORS]
y_test = df_test[TARGET]

groups = df_train["subject_id"]

baseline_rate = y_test.mean()
print("\nbaseline rate:", round(baseline_rate, 4))

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=PREDICTORS)
X_test = pd.DataFrame(imputer.transform(X_test), columns=PREDICTORS)

joblib.dump(imputer, "imputer_reinsertion.pkl")

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    subsample=0.8,
    random_state=42
)

pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
group_kfold = GroupKFold(n_splits=5)

cv_results = cross_validate(
    pipeline, X_train, y_train, cv=group_kfold, groups=groups,
    scoring=["roc_auc", "average_precision", "neg_brier_score"]
)

print("ROC-AUC:", round(cv_results["test_roc_auc"].mean(), 4))
print("AUPRC:  ", round(cv_results["test_average_precision"].mean(), 4))
print("Brier:  ", round((-cv_results["test_neg_brier_score"]).mean(), 4))

calibrated_model = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)
joblib.dump(calibrated_model, "model_reinsertion.pkl")

y_prob = calibrated_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
auprc = average_precision_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n" + "=" * 55)
print("PROBABILITY MODEL EVALUATION")
print("=" * 55)
print("ROC-AUC:", round(roc_auc, 4))
print("AUPRC:  ", round(auprc, 4))
print("Brier:  ", round(brier, 4))


# ==============================================================================
# PART 1b - LOAD THE CAUTI MODEL (drives the risk-threshold removal decision)
# ==============================================================================
#
# The removal decision is driven by predicted CAUTI risk, not reinsertion
# risk — CAUTI risk is defined while the catheter is still in place, which
# is exactly when the removal decision needs to be made. Assumes
# model_cauti.pkl / imputer_cauti.pkl already exist from your CAUTI script.

CAUTI_PREDICTORS = [
    "age",
    "sex_M",
    "episode_index",
    "interval_hours",
    "periods_in_state",
    "itemid_229371__last",
    "itemid_220739__last",
    "itemid_223901__last",
    "itemid_223900__last",
    "itemid_220045__mean",
    "itemid_220052__mean",
    "itemid_220210__mean",
    "itemid_223762__mean",
    "itemid_220615__last",
    "itemid_225624__last",
    "itemid_227471__last",
    "itemid_220546__last",
    "itemid_229357__last",
    "itemid_220228__last",
    "itemid_227457__last",
    "itemid_220645__last",
    "itemid_227442__last",
    "itemid_220602__last",
    "itemid_220621__last",
    "itemid_227456__last",
    "itemid_220644__last",
    "itemid_220587__last",
    "itemid_225690__last",
    "itemid_225612__last",
    "itemid_227073__last",
    "itemid_220235__last",
    "itemid_220224__last",
    "itemid_220227__last",
    "itemid_220051__mean",
    "itemid_220050__mean",
]

cauti_imputer = joblib.load("imputer_cauti.pkl")
cauti_model = joblib.load("model_cauti.pkl")


# ==============================================================================
# PART 2 - BUILD G FORMULA DATASET AND COUNTERFACTUAL TRAJECTORIES
# ==============================================================================

df_sim = df_full[df_full["split"] == "test"].copy()
df_sim = df_sim.sort_values(
    ["subject_id", "episode_index", "periods_in_state"]
).reset_index(drop=True)

print("\nSimulation dataset created.")
print("Rows:    ", len(df_sim))
print("Patients:", df_sim["subject_id"].nunique())

df_sim_atrisk = df_sim[df_sim["at_risk_reinsertion"] == 1].copy()
print("At-risk rows available for policy simulation:", len(df_sim_atrisk))

EPISODE_KEY = ["subject_id", "hadm_id", "stay_id", "episode_index"]
TIME_KEY = "interval_hours"

PREDICTORS_CF = [
    "periods_in_state_cf" if col == "periods_in_state" else col
    for col in PREDICTORS
]

# ------------------------------------------------------------------
# Precompute predicted CAUTI probability for every row where the
# catheter is still in place (at_risk_cauti == 1), ONCE, globally.
# This is the REAL predicted CAUTI risk at that observed moment — not a
# hypothetical — since it reflects the patient's actual current clinical
# state while still catheterized, which is exactly when the removal
# decision needs to be made.
# ------------------------------------------------------------------
cauti_scored_mask = df_sim["at_risk_cauti"] == 1
X_cauti = pd.DataFrame(
    cauti_imputer.transform(df_sim.loc[cauti_scored_mask, CAUTI_PREDICTORS]),
    columns=CAUTI_PREDICTORS,
    index=df_sim.loc[cauti_scored_mask].index
)
df_sim["predicted_cauti_risk"] = np.nan
df_sim.loc[cauti_scored_mask, "predicted_cauti_risk"] = (
    cauti_model.predict_proba(X_cauti)[:, 1]
)

# Clinical-stability criteria
STABILITY_COLS = {
    "temp": "itemid_223762__mean",
    "hr": "itemid_220045__mean",
    "sbp": "itemid_220050__mean",
    "o2sat": "itemid_220227__last",
}

STABILITY_THRESHOLDS = {
    "temp_max": 38.0,
    "hr_max": 100.0,
    "sbp_min": 100.0,
    "o2sat_min": 94.0,
}

POLICIES = {
    "Observed Practice": {"type": "natural"},

    "Remove day 1": {"type": "day", "day": 1},
    "Remove day 2": {"type": "day", "day": 2},
    "Remove day 3": {"type": "day", "day": 3},
    "Remove day 4": {"type": "day", "day": 4},
    "Remove day 5": {"type": "day", "day": 5},
    "Remove day 6": {"type": "day", "day": 6},

    "Clinical stability": {"type": "stability"},
    "Clinical stability (>= Day 3)": {"type": "stability_min_day", "min_day": 3},

    "Hybrid (stability OR day 4 cap)": {"type": "hybrid", "day_cap": 4},
    "Hybrid (stability OR day 5 cap)": {"type": "hybrid", "day_cap": 5},
    "Hybrid (stability OR day 6 cap)": {"type": "hybrid", "day_cap": 6},

    # Remove as soon as predicted CAUTI risk reaches X% — reinsertion
    # risk is then measured as the DOWNSTREAM CONSEQUENCE of that choice.
    "Remove when CAUTI risk >= 1%": {"type": "cauti_risk_threshold", "threshold": 0.01},
    "Remove when CAUTI risk >= 2%": {"type": "cauti_risk_threshold", "threshold": 0.02},
    "Remove when CAUTI risk >= 5%": {"type": "cauti_risk_threshold", "threshold": 0.05},
    "Remove when CAUTI risk >= 10%": {"type": "cauti_risk_threshold", "threshold": 0.10},
    "Remove when CAUTI risk >= 25%": {"type": "cauti_risk_threshold", "threshold": 0.25},
    "Remove when CAUTI risk >= 50%": {"type": "cauti_risk_threshold", "threshold": 0.50},
}


def create_counterfactual_patient(episode_rows, policy):
    """
    Decide removed_in_period for one episode under a given policy.
    "day": force removal once periods_in_state reaches the threshold.
    "stability": remove at first period all 4 vitals meet criteria.
    "stability_min_day": stability AND a minimum elapsed day.
    "hybrid": remove at whichever of (stability, day cap) comes first.
    "cauti_risk_threshold": remove at first period predicted CAUTI risk
        reaches the threshold. No model call here — predicted_cauti_risk
        was already computed once, globally, above, so this is pure
        bookkeeping, exactly like every other policy type.
    """
    patient_cf = episode_rows.sort_values(TIME_KEY).copy()
    ptype = policy["type"]

    patient_cf["removed_in_period"] = 0

    if ptype == "natural":
        patient_cf["reinsertion_state_cf"] = np.where(
            patient_cf["at_risk_reinsertion"] == 1, "AT_RISK", "NOT_AT_RISK"
        )
        patient_cf["periods_in_state_cf"] = patient_cf["periods_in_state"]
        return patient_cf

    removed = False

    for idx in patient_cf.index:
        if removed:
            continue

        day = patient_cf.loc[idx, "periods_in_state"]

        if ptype == "day":
            trigger = day >= policy["day"]

        elif ptype == "cauti_risk_threshold":
            cauti_risk = patient_cf.loc[idx, "predicted_cauti_risk"]
            trigger = pd.notna(cauti_risk) and (cauti_risk >= policy["threshold"])

        else:  # "stability", "stability_min_day", or "hybrid"
            temp = patient_cf.loc[idx, STABILITY_COLS["temp"]]
            hr = patient_cf.loc[idx, STABILITY_COLS["hr"]]
            sbp = patient_cf.loc[idx, STABILITY_COLS["sbp"]]
            o2sat = patient_cf.loc[idx, STABILITY_COLS["o2sat"]]

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
                trigger = is_stable
            elif ptype == "stability_min_day":
                trigger = is_stable and (day >= policy["min_day"])
            else:  # hybrid
                trigger = is_stable or (day >= policy["day_cap"])

        if trigger:
            patient_cf.loc[idx, "removed_in_period"] = 1
            removed = True

    return patient_cf


def update_reinsertion_state(patient_cf):
    """Unchanged — bookkeeping only, no model calls."""
    at_risk = False
    state = []
    periods_cf = []
    periods_since_removal = 0

    for index, row in patient_cf.iterrows():
        if row["removed_in_period"] == 1:
            at_risk = True
            periods_since_removal = 0

        state.append("AT_RISK" if at_risk else "NOT_AT_RISK")

        if at_risk:
            periods_cf.append(periods_since_removal)
            periods_since_removal += 1
        else:
            periods_cf.append(row["periods_in_state"])

    patient_cf["reinsertion_state_cf"] = state
    patient_cf["periods_in_state_cf"] = periods_cf

    return patient_cf


def cumulative_risk(patient_cf):
    p = patient_cf["predicted_probability"].values
    return 1 - np.prod(1 - p)


# ==============================================================================
# PART 3 - IMPLEMENT THE G FORMULA AND RESULTS
# ==============================================================================
#
# cauti_risk_threshold makes its removal decision inside
# create_counterfactual_patient, exactly like every other policy type, so
# it flows through update_reinsertion_state and the batched reinsertion
# prediction the same way — no separate post-hoc block needed.

results = []

for policy_name, policy in POLICIES.items():

    episode_frames = []

    for key, episode in df_sim.groupby(EPISODE_KEY):
        cf = create_counterfactual_patient(episode, policy)

        if len(cf) == 0:
            continue

        if policy["type"] != "natural":
            cf = update_reinsertion_state(cf)

        episode_frames.append(cf)

    if len(episode_frames) == 0:
        results.append({"Policy": policy_name, "Mean Risk": 0.0})
        continue

    all_cf = pd.concat(episode_frames)
    all_cf["predicted_probability"] = 0.0

    at_risk_mask = all_cf["reinsertion_state_cf"] == "AT_RISK"
    at_risk_rows = all_cf.loc[at_risk_mask]

    if len(at_risk_rows) > 0:
        X = at_risk_rows[PREDICTORS_CF].copy()
        X.columns = PREDICTORS

        X_imputed = pd.DataFrame(
            imputer.transform(X), columns=PREDICTORS, index=X.index
        )

        probs = calibrated_model.predict_proba(X_imputed)[:, 1]
        all_cf.loc[at_risk_rows.index, "predicted_probability"] = probs

    episode_risks = []
    for key, group in all_cf.groupby(EPISODE_KEY):
        episode_risks.append(cumulative_risk(group))

    results.append({"Policy": policy_name, "Mean Risk": np.mean(episode_risks)})

policy_results = pd.DataFrame(results)
policy_results["Mean Risk (%)"] = (policy_results["Mean Risk"] * 100).round(3)

print("\n" + "=" * 70)
print("G-FORMULA REINSERTION POLICY RISK ESTIMATES (cumulative per-patient risk)")
print("=" * 70)
print(policy_results.to_string(index=False))