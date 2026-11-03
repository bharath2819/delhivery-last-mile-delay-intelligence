
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    NOISE_COLUMNS,
    build_features,
    clean_shipments,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def save_csv(df, filename):
    df.to_csv(OUTPUT_DIR / filename, index=False, float_format="%.4f")


def main():
    # 1. Load source data.
    raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
    hubs = pd.read_csv(DATA_DIR / "hubs.csv")

    # 2. Capture data-quality checks before cleaning.
    duplicate_rows_removed = int(raw.duplicated().sum())
    missing_distance_before = int(raw["distance_km"].isna().sum())
    missing_weight_before = int(
    raw.drop_duplicates()["parcel_weight_kg"].isna().sum())
    missing_value_before = int(raw["declared_value"].isna().sum())
    missing_payment_before = int(raw["payment_mode"].isna().sum())
    sentinel_weight_rows = int((raw["parcel_weight_kg"] > 100).sum())
    service_spellings_before = int(raw["service_type"].nunique())

    # 3. Clean historical shipment data.
    clean, medians = clean_shipments(raw)

    service_spellings_after = int(clean["service_type"].nunique())

    # Save imputation medians for future prediction.
    median_df = pd.DataFrame(
        [
            {"column": column, "median": value}
            for column, value in medians.items()
        ]
    )
    median_df.to_csv(MODEL_DIR / "impute_values.csv", index=False, float_format="%.4f")

    # 4. Build features by merging hub information.
    model_df = build_features(clean, hubs)

    # 5. Create the target only after feature construction.
    model_df["is_delayed"] = (
        model_df["delivery_hours"] > model_df["promised_hours"]
    ).astype(int)

    # 6. Check feature and target null values.
    nulls_in_features_and_target = int(
        model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    )

    # 7. Save the modelling table.
    save_csv(model_df, "modelling_table.csv")
    model_df.to_csv(DATA_DIR / "modelling_table.csv", index=False, float_format="%.4f")

    # 8. Calculate delivery-hour outliers using the IQR rule.
    q1 = float(model_df["delivery_hours"].quantile(0.25))
    q3 = float(model_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = int(
        (
            (model_df["delivery_hours"] < lower_bound)
            | (model_df["delivery_hours"] > upper_bound)
        ).sum()
    )

    # 9. Calculate delay rate.
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = float(model_df["is_delayed"].mean())

    # 10. Create the summary checks table.
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", len(clean)),
        ("missing_distance_km_before", missing_distance_before),
        ("missing_parcel_weight_before", missing_weight_before),
        ("missing_declared_value_before", missing_value_before),
        ("missing_payment_mode_before", missing_payment_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(model_df)),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower_bound),
        ("hours_upper_bound", upper_bound),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", float(model_df["declared_value_inr"].mean())),
        ("declared_value_median_inr", float(model_df["declared_value_inr"].median())),
        ("log_declared_value_mean", float(model_df["log_declared_value"].mean())),
        ("log_declared_value_median", float(model_df["log_declared_value"].median())),
    ]

    checks_df = pd.DataFrame(checks, columns=["check", "value"])
    save_csv(checks_df, "data_quality_checks.csv")

    # 11. Delay rate by service type.
    service_summary = (
        model_df.groupby("service_type", observed=False)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )

    save_csv(service_summary, "delay_by_service.csv")

    # 12. Delay rate by hub tier.
    hub_summary = (
        model_df.groupby("hub_tier", observed=False)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )

    save_csv(hub_summary, "delay_by_hub_tier.csv")

    # 13. Delay rate by distance band.
    distance_bins = [0, 200, 400, 600, 2000]
    distance_labels = ["0-200", "201-400", "401-600", "601+"]

    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        right=True,
        include_lowest=True,
    )

    distance_summary = (
        model_df.groupby("distance_band", observed=False)
        .agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
    )

    save_csv(distance_summary, "delay_by_distance_band.csv")

    # 14. Numeric correlation matrix.
    numeric_df = model_df.select_dtypes(include=np.number)
    correlation = numeric_df.corr()
    correlation.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv",
        index=True,
        float_format="%.4f",
        index_label="feature",
    )

    # 15. Print important results for validation.
    print("Part A completed successfully.")
    print(f"Raw rows: {len(raw)}")
    print(f"Rows after cleaning: {len(clean)}")
    print(f"Delayed shipments: {delayed_count}")
    print(f"Delay rate: {delay_rate:.4f}")
    print(f"Nulls in features and target: {nulls_in_features_and_target}")
    print(f"Correlation(delivery_hours, promised_hours): {correlation.loc['delivery_hours', 'promised_hours']:.3f}")
    print(f"Correlation(delivery_hours, distance_km): {correlation.loc['delivery_hours', 'distance_km']:.3f}")
    print(f"Correlation(is_delayed, distance_km): {correlation.loc['is_delayed', 'distance_km']:.3f}")
    print("\nOutput files saved in:", OUTPUT_DIR)
    print("Imputation values saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
