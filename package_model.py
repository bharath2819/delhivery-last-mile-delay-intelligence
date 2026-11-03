
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# ---------------------------------------------------------
# Paths and fixed configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load the selected Part C configuration
# ---------------------------------------------------------

config = pd.read_csv(OUTPUT_DIR / "part_c_final_config.csv")

if len(config) != 1:
    raise ValueError("Expected exactly one row in part_c_final_config.csv")

config_row = config.iloc[0]

selector_k = int(config_row["selector_k"])
n_estimators = int(config_row["n_estimators"])
min_samples_leaf = int(config_row["min_samples_leaf"])

max_depth_value = str(config_row["max_depth"]).strip().lower()
max_depth = None if max_depth_value == "unlimited" else int(max_depth_value)

threshold = float(config_row["threshold"])

if selector_k != 7:
    raise ValueError(f"Unexpected selector_k: {selector_k}")

if n_estimators != 100 or max_depth is not None or min_samples_leaf != 1:
    raise ValueError("Part C model configuration does not match the selected model")


# ---------------------------------------------------------
# Rebuild the exact historical imputation medians
# ---------------------------------------------------------

raw_shipments = pd.read_csv(DATA_DIR / "raw_shipments.csv")

_, medians = clean_shipments(raw_shipments)

impute_df = pd.DataFrame([
    {"column": column, "median": value}
    for column, value in medians.items()
])

impute_df.to_csv(
    MODEL_DIR / "impute_values.csv",
    index=False,
    float_format="%.10f",
)


# ---------------------------------------------------------
# Rebuild the exact Part C stratified split
# ---------------------------------------------------------

df = pd.read_csv(DATA_DIR / "modelling_table.csv")

X = df[FEATURES].copy()
y = df["is_delayed"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum()),
) == (195, 42, 41)


# ---------------------------------------------------------
# Build and fit the final classification pipeline
# ---------------------------------------------------------

final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
        ),
    ),
])

final_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Save the model and supporting files
# ---------------------------------------------------------

model_path = MODEL_DIR / "delay_model.pkl"

joblib.dump(final_model, model_path)

pd.DataFrame({
    "threshold": [threshold],
}).to_csv(
    MODEL_DIR / "delay_threshold.csv",
    index=False,
    float_format="%.4f",
)

pd.DataFrame({
    "feature": FEATURES,
}).to_csv(
    MODEL_DIR / "feature_order.csv",
    index=False,
)

# Confirm that the saved feature order contains all 17 features.
assert len(FEATURES) == 17
assert len(set(FEATURES)) == 17


# ---------------------------------------------------------
# Reload the saved model and compare test probabilities
# ---------------------------------------------------------

reloaded_model = joblib.load(model_path)

original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_probability_difference = float(
    np.max(
        np.abs(
            original_probabilities - reloaded_probabilities
        )
    )
)

reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test),
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_difference,
    },
    {
        "metric": "threshold",
        "value": threshold,
    },
])

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False,
    float_format="%.12f",
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert max_probability_difference <= 1e-9, (
    "Reloaded model probabilities do not match the original model."
)

assert len(pd.read_csv(MODEL_DIR / "feature_order.csv")) == 17
assert len(pd.read_csv(MODEL_DIR / "impute_values.csv")) == 3
assert len(pd.read_csv(MODEL_DIR / "delay_threshold.csv")) == 1

print("\nPART D — MODEL PACKAGING")
print("------------------------")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows: {len(X_test)}")
print(f"Selected features: {selector_k}")
print(f"Random forest trees: {n_estimators}")
print(f"Maximum depth: {max_depth}")
print(f"Minimum samples per leaf: {min_samples_leaf}")
print(f"Alert threshold: {threshold:.4f}")

print("\nIMPUTATION MEDIANS")
print("------------------")
print(impute_df.to_string(index=False))

print("\nMODEL RELOAD CHECK")
print("------------------")
print(reload_check.to_string(index=False))

print("\nSaved files:")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")
print("outputs/part_d_reload_check.csv")

print("\nModel packaging completed successfully.")
