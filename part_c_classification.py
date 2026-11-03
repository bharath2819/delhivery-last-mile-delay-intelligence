from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Return confusion-matrix cells and classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


def save_csv(df, filename):
    """Save an output CSV without an index."""
    df.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        float_format="%.4f",
    )


# ---------------------------------------------------------
# Load data and perform the stratified split
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

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

print("\nPART C — STRATIFIED SPLIT")
print("-------------------------")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")
print(
    "Delayed parcels:",
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum()),
)

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum()),
) == (195, 42, 41)


# ---------------------------------------------------------
# Task 2 — Baseline and three classifiers
# ---------------------------------------------------------

model_results = []

# Blind baseline: predict every parcel as on time.
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_metrics(y_val, blind_predictions)
model_results.append({
    "model": "blind_all_on_time",
    **blind_metrics,
})

# Logistic regression: scaling is fitted on training rows.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

model_results.append({
    "model": "logistic_regression",
    **calculate_metrics(y_val, logistic_predictions),
})

# Find the decision-tree depth with the highest validation F1.
tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=RANDOM_STATE,
    )

    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)

    tree_depth_results.append({
        "depth": depth,
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "model": tree,
    })

best_tree_result = sorted(
    tree_depth_results,
    key=lambda row: (-row["f1"], row["depth"]),
)[0]

best_tree_depth = best_tree_result["depth"]
best_tree = best_tree_result["model"]

model_results.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **calculate_metrics(y_val, best_tree.predict(X_val)),
})

# Default random forest using all 17 features.
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)

default_forest.fit(X_train, y_train)
default_forest_predictions = default_forest.predict(X_val)

model_results.append({
    "model": "random_forest_default",
    **calculate_metrics(y_val, default_forest_predictions),
})

models_df = pd.DataFrame(model_results)

model_columns = [
    "model", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

models_df = models_df[model_columns]
save_csv(models_df, "part_c_models.csv")


# ---------------------------------------------------------
# Task 3 — Feature scores and SelectKBest experiment
# ---------------------------------------------------------

# Random forest importance using all 17 training features.
importance_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)

importance_forest.fit(X_train, y_train)

# F-scores using all 17 training features.
all_features_selector = SelectKBest(
    score_func=f_classif,
    k="all",
)

all_features_selector.fit(X_train, y_train)

f_scores = all_features_selector.scores_
rf_importances = importance_forest.feature_importances_

# Rank highest F-score as rank 1.
feature_score_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [
        feature in NOISE_COLUMNS for feature in FEATURES
    ],
})

feature_score_df["f_rank"] = (
    feature_score_df["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_score_df = feature_score_df[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_score_df = feature_score_df.sort_values(
    "f_rank"
).reset_index(drop=True)

save_csv(feature_score_df, "part_c_feature_scores.csv")


# Evaluate different feature counts.
selectk_results = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    selectk_model.fit(X_train, y_train)
    predictions = selectk_model.predict(X_val)

    selectk_results.append({
        "k": k,
        "val_f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    })

selectk_df = pd.DataFrame(selectk_results)
save_csv(selectk_df, "part_c_selectk.csv")


# ---------------------------------------------------------
# Task 4 — Grid search and randomized search using k=7
# ---------------------------------------------------------

# Fit feature selection on training rows only.
selector_for_tuning = SelectKBest(
    score_func=f_classif,
    k=7,
)

X_train_7 = selector_for_tuning.fit_transform(X_train, y_train)
X_val_7 = selector_for_tuning.transform(X_val)

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

base_forest = RandomForestClassifier(
    random_state=RANDOM_STATE
)

grid_search = GridSearchCV(
    estimator=base_forest,
    param_grid=parameter_grid,
    cv=5,
)

grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.best_estimator_.predict(X_val_7)
grid_val_f1 = f1_score(
    y_val, grid_predictions, zero_division=0
)

randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE
    ),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=RANDOM_STATE,
)

randomized_search.fit(X_train_7, y_train)

randomized_predictions = (
    randomized_search.best_estimator_.predict(X_val_7)
)

randomized_val_f1 = f1_score(
    y_val, randomized_predictions, zero_division=0
)


# Untuned forest with the same seven selected features.
default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)

default_forest_7.fit(X_train_7, y_train)

default_7_predictions = default_forest_7.predict(X_val_7)

default_7_val_f1 = f1_score(
    y_val, default_7_predictions, zero_division=0
)


tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": json.dumps(
            grid_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1,
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": json.dumps(
            randomized_search.best_params_,
            sort_keys=True,
        ),
        "best_cv_accuracy": randomized_search.best_score_,
        "val_f1": randomized_val_f1,
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_7_val_f1,
    },
]

tuning_df = pd.DataFrame(tuning_results)

tuning_columns = [
    "search",
    "candidates",
    "cv_folds",
    "total_fits",
    "best_params",
    "best_cv_accuracy",
    "val_f1",
]

tuning_df = tuning_df[tuning_columns]
save_csv(tuning_df, "part_c_tuning.csv")


# ---------------------------------------------------------
# Task 5 — Choose final configuration from validation F1
# ---------------------------------------------------------

tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

winning_row = sorted(
    tuning_results,
    key=lambda row: (
        -row["val_f1"],
        tie_priority[row["search"]],
    ),
)[0]

chosen_from = winning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = randomized_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

# Build the required final pipeline and fit on training rows only.
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    (
        "model",
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            **chosen_params,
        ),
    ),
])

final_model.fit(X_train, y_train)


# ---------------------------------------------------------
# Task 6 — Choose alert threshold using validation recall
# ---------------------------------------------------------

validation_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in THRESHOLDS:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        **calculate_metrics(y_val, predictions),
    })

thresholds_df = pd.DataFrame(threshold_results)

threshold_columns = [
    "threshold", "TN", "FP", "FN", "TP",
    "accuracy", "precision", "recall", "f1",
]

thresholds_df = thresholds_df[threshold_columns]
save_csv(thresholds_df, "part_c_thresholds.csv")

eligible_thresholds = [
    row["threshold"]
    for row in threshold_results
    if row["recall"] >= 0.80
]

chosen_threshold = (
    max(eligible_thresholds)
    if eligible_thresholds
    else 0.1
)


# ---------------------------------------------------------
# Task 7 — One final test evaluation
# ---------------------------------------------------------

# Do not use test rows for any model or threshold selection.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

final_test_metrics = calculate_metrics(
    y_test,
    test_predictions,
)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **final_test_metrics,
}])

save_csv(final_test_df, "part_c_final_test.csv")


# Save the final configuration.
max_depth_value = chosen_params["max_depth"]

final_config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "unlimited"
        if max_depth_value is None
        else max_depth_value
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])

save_csv(final_config_df, "part_c_final_config.csv")


# ---------------------------------------------------------
# Final checks and summary
# ---------------------------------------------------------

noise_importance_share = feature_score_df.loc[
    feature_score_df["is_noise"], "rf_importance"
].sum()

top_seven = set(
    feature_score_df.nsmallest(7, "f_rank")["feature"]
)

noise_outside_top_seven = all(
    feature not in top_seven
    for feature in NOISE_COLUMNS
)

print("\nPART C — MODEL RESULTS")
print("----------------------")
print(models_df.to_string(index=False))

print("\nFEATURE IMPORTANCE")
print("------------------")
print(feature_score_df.to_string(index=False))
print(f"\nCombined noise importance: {noise_importance_share:.4f}")
print(f"All noise columns outside top 7: {noise_outside_top_seven}")

print("\nSELECTKBEST RESULTS")
print("-------------------")
print(selectk_df.to_string(index=False))

print("\nTUNING RESULTS")
print("--------------")
print(tuning_df.to_string(index=False))

print(f"\nChosen configuration: {chosen_from}")
print(f"Chosen parameters: {chosen_params}")

print("\nTHRESHOLD RESULTS")
print("-----------------")
print(thresholds_df.to_string(index=False))
print(f"\nChosen alert threshold: {chosen_threshold}")

print("\nFINAL TEST RESULTS")
print("------------------")
print(final_test_df.to_string(index=False))

print("\nPart C completed successfully.")