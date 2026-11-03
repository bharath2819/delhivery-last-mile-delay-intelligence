
import numpy as np
import pandas as pd


PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean shipment data and return the cleaned data with imputation medians."""

    df = raw.copy()

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Standardize service types.
    df["service_type"] = df["service_type"].astype("string").str.strip().str.title()

    # Fill missing payment modes.
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Convert declared value into a numeric INR column.
    df["declared_value_inr"] = pd.to_numeric(
    df["declared_value"].astype("string")
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip(),
    errors="coerce",
).astype("float64")
    df = df.drop(columns=["declared_value"])

    # Convert dispatch time to datetime.
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # Treat weights above 100 kg as invalid sentinel values.
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Calculate historical medians or use supplied medians.
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            column: float(df[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # Fill missing numeric values.
    for column in median_columns:
        df[column] = df[column].fillna(medians[column])

    return df, medians


def build_features(clean, hubs):
    """Merge hub details and create model-ready features."""

    df = clean.copy()

    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    df = df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    # Service-level delivery promise.
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # Binary and time-based features.
    df["is_cod"] = (
        df["payment_mode"].astype("string").str.strip().str.upper() == "COD"
    ).astype(int)

    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # Log transformation of declared value.
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
