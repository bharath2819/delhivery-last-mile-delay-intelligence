
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_DIR / "modelling_table.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")


# ---------------------------------------------------------
# Task 1 — Build hub profile
# ---------------------------------------------------------

hub_profile = (
    df.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

hub_profile.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# Task 2 — Scale and cluster the hubs
# ---------------------------------------------------------

scaler = StandardScaler()

scaled_profiles = scaler.fit_transform(
    hub_profile[profile_columns]
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

cluster_labels = kmeans.fit_predict(scaled_profiles)

hub_profile["cluster"] = cluster_labels


# ---------------------------------------------------------
# Task 3 — Separation evidence for K = 2, 3, 4, 5
# ---------------------------------------------------------

distance_matrix = pairwise_distances(
    scaled_profiles,
    metric="euclidean",
)

upper_triangle = np.triu_indices(
    len(hub_profile),
    k=1,
)

all_distances = distance_matrix[upper_triangle]

separation_results = []

for k in [2, 3, 4, 5]:
    candidate = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = candidate.fit_predict(scaled_profiles)

    row_indices, col_indices = upper_triangle

    same_cluster = labels[row_indices] == labels[col_indices]
    different_cluster = ~same_cluster

    within_mean = all_distances[same_cluster].mean()
    across_mean = all_distances[different_cluster].mean()

    ratio = across_mean / within_mean

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True,
    )

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        ),
    })

separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# Task 4 — Give the three clusters business names
# ---------------------------------------------------------

cluster_delay_rates = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

cluster_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate",
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(cluster_names)
)


# ---------------------------------------------------------
# Task 5 — Hub segments with city and tier
# ---------------------------------------------------------

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left",
    validate="one_to_one",
)

hub_segments = hub_segments.sort_values(
    "hub_id"
).reset_index(drop=True)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False,
)


# ---------------------------------------------------------
# Task 6 — Segment-level profile
# ---------------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2,
}

segment_profile["_order"] = (
    segment_profile["segment"].map(segment_order)
)

segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False,
    float_format="%.4f",
)


# ---------------------------------------------------------
# Task 7 — Find tier versus segment mismatches
# ---------------------------------------------------------

tier_to_expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(tier_to_expected_segment)
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False,
)


# ---------------------------------------------------------
# Print verification results
# ---------------------------------------------------------

print("\nPART D — HUB PROFILE")
print("--------------------")
print(f"Hub count: {len(hub_profile)}")
print(
    "Shipment counts:",
    sorted(hub_profile["shipments"].unique().tolist()),
)

print("\nREFERENCE HUB CHECKS")
print("--------------------")

for hub_id in ["HUB01", "HUB13", "HUB17"]:
    print(
        hub_profile.loc[
            hub_profile["hub_id"] == hub_id,
            [
                "hub_id",
                "shipments",
                "mean_delivery_hours",
                "delay_rate",
                "mean_distance_km",
                "mean_n_stops",
            ],
        ].to_string(index=False)
    )

print("\nK-MEANS SEPARATION")
print("------------------")
print(separation_df.to_string(index=False))

print("\nSEGMENT PROFILE")
print("---------------")
print(segment_profile.to_string(index=False))

print("\nTIER VERSUS SEGMENT MISMATCHES")
print("------------------------------")
print(mismatches.to_string(index=False))

print("\nPart D segmentation completed successfully.")
