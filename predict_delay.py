
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import build_features, clean_shipments


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load new shipments and supporting files
# ---------------------------------------------------------

new_shipments = pd.read_csv(DATA_DIR / "new_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

model = joblib.load(MODEL_DIR / "delay_model.pkl")

threshold_df = pd.read_csv(MODEL_DIR / "delay_threshold.csv")
threshold = float(threshold_df.loc[0, "threshold"])

feature_order_df = pd.read_csv(MODEL_DIR / "feature_order.csv")
feature_order = feature_order_df["feature"].tolist()

impute_df = pd.read_csv(MODEL_DIR / "impute_values.csv")

medians = dict(
    zip(
        impute_df["column"],
        impute_df["median"],
    )
)

hub_segments = pd.read_csv(OUTPUT_DIR / "hub_segments.csv")


# ---------------------------------------------------------
# Validate supporting files
# ---------------------------------------------------------

if len(new_shipments) != 48:
    raise ValueError(
        f"Expected 48 new shipments, found {len(new_shipments)}"
    )

if len(feature_order) != 17 or len(set(feature_order)) != 17:
    raise ValueError("Feature order must contain 17 unique features.")

if set(medians) != {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
}:
    raise ValueError("Imputation file does not contain the three required medians.")

if len(threshold_df) != 1:
    raise ValueError("Threshold file must contain exactly one row.")

if hub_segments["hub_id"].duplicated().any():
    raise ValueError("Hub segment file contains duplicate hub IDs.")


# ---------------------------------------------------------
# Clean new shipments using historical training medians
# ---------------------------------------------------------

cleaned_shipments, used_medians = clean_shipments(
    new_shipments,
    medians=medians,
)

# The supplied historical medians must be used unchanged.
assert used_medians == medians


# ---------------------------------------------------------
# Build features and preserve the saved feature order
# ---------------------------------------------------------

featured_shipments = build_features(cleaned_shipments, hubs)

missing_features = [
    feature for feature in feature_order
    if feature not in featured_shipments.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X_new = featured_shipments[feature_order].copy()

if X_new.isna().any().any():
    raise ValueError("New shipment feature matrix contains missing values.")


# ---------------------------------------------------------
# Predict delay probabilities
# ---------------------------------------------------------

raw_probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities to four decimal places for the output.
rounded_probabilities = np.round(raw_probabilities, 4)

# Use the unrounded probability for threshold comparison.
flagged = raw_probabilities >= threshold


# ---------------------------------------------------------
# Attach hub segments
# ---------------------------------------------------------

results = featured_shipments[
    ["shipment_id", "hub_id"]
].copy()

results["segment"] = results["hub_id"].map(
    hub_segments.set_index("hub_id")["segment"]
)

if results["segment"].isna().any():
    missing_hubs = results.loc[
        results["segment"].isna(), "hub_id"
    ].unique().tolist()

    raise ValueError(
        f"Missing segment mapping for hubs: {missing_hubs}"
    )


# ---------------------------------------------------------
# Assign operational actions
# ---------------------------------------------------------

results["delay_probability"] = rounded_probabilities
results["flagged"] = flagged

results["action"] = np.select(
    [
        results["flagged"] & (results["segment"] == "Escalate"),
        results["flagged"],
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call",
    ],
    default="No action",
)


# ---------------------------------------------------------
# Save the final output
# ---------------------------------------------------------

results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]

if len(results) != 48:
    raise ValueError("Final output must contain exactly 48 rows.")

if results["shipment_id"].duplicated().any():
    raise ValueError("Duplicate shipment IDs found in final output.")

if results.isna().any().any():
    raise ValueError("Final output contains missing values.")

output_path = OUTPUT_DIR / "new_shipment_actions.csv"

results.to_csv(
    output_path,
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# Print summary and validation checks
# ---------------------------------------------------------

print("\nNEW SHIPMENT DELAY PREDICTIONS")
print("------------------------------")
print(f"Input shipments: {len(new_shipments)}")
print(f"Output shipments: {len(results)}")
print(f"Alert threshold: {threshold:.4f}")

print("\nACTION COUNTS")
print("-------------")
print(results["action"].value_counts().to_string())

print("\nFLAGGED SHIPMENTS BY SEGMENT")
print("----------------------------")
print(
    results.groupby(
        ["segment", "flagged"],
        observed=True,
    ).size().to_string()
)

print("\nOUTPUT PREVIEW")
print("--------------")
print(results.head(10).to_string(index=False))

print("\nVALIDATION")
print("----------")
print(f"Unique shipment IDs: {results['shipment_id'].nunique()}")
print(f"Missing values: {int(results.isna().sum().sum())}")
print(f"Flagged shipments: {int(results['flagged'].sum())}")
print(
    "Hub manager escalations:",
    int((results["action"] == "Escalate to hub manager").sum()),
)

print(f"\nSaved: {output_path}")
print("\nNew shipment prediction completed successfully.")
