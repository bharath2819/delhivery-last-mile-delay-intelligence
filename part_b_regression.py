
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def main():
    # 1. Load the modelling table.
    df = pd.read_csv(DATA_DIR / "modelling_table.csv")

    X = df[FEATURES]
    y = df["delivery_hours"]

    # 2. Split into training, validation and test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    # 3. Fit the scaler on training rows only.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(
        np.abs(X_train_scaled.mean(axis=0)).max()
    )

    test_scaled_max_abs_mean = float(
        np.abs(X_test_scaled.mean(axis=0)).max()
    )

    checks = pd.DataFrame(
        [
            ("train_rows", len(X_train)),
            ("val_rows", len(X_val)),
            ("test_rows", len(X_test)),
            ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
            ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
        ],
        columns=["metric", "value"],
    )

    checks.to_csv(
        OUTPUT_DIR / "part_b_checks.csv",
        index=False,
        float_format="%.4f",
    )

    # 4. Mean baseline.
    baseline_prediction = np.full(
        len(y_val),
        y_train.mean(),
    )

    metrics_rows = []

    baseline_metrics = calculate_metrics(y_val, baseline_prediction)

    metrics_rows.append({
        "model": "mean_baseline",
        "split": "val",
        **baseline_metrics,
    })

    # 5. Decision tree depth diagnostic.
    depths = [1, 2, 3, 4, 6, 8, 12, None]

    depth_rows = []
    depth_models = {}

    for depth in depths:
        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42,
        )

        tree.fit(X_train, y_train)

        train_prediction = tree.predict(X_train)
        val_prediction = tree.predict(X_val)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_prediction)
        )

        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_prediction)
        )

        depth_name = "unlimited" if depth is None else depth

        depth_rows.append({
            "max_depth": depth_name,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })

        depth_models[depth_name] = tree

    depth_df = pd.DataFrame(depth_rows)

    depth_df.to_csv(
        OUTPUT_DIR / "part_b_tree_depth.csv",
        index=False,
        float_format="%.4f",
    )

    # Select the best tree depth.
    best_depth_row = min(
        depth_rows,
        key=lambda row: (
            row["val_rmse"],
            float("inf") if row["max_depth"] == "unlimited"
            else row["max_depth"],
        ),
    )

    best_depth = best_depth_row["max_depth"]
    best_tree = depth_models[best_depth]

    # 6. Define the four candidate regression models.
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),

        "decision_tree": best_tree,

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    validation_rmse = {}

    # Fit candidates on training data and score on validation data.
    for name, model in models.items():
        model.fit(X_train, y_train)

        val_prediction = model.predict(X_val)

        model_metrics = calculate_metrics(
            y_val,
            val_prediction,
        )

        validation_rmse[name] = model_metrics["RMSE"]

        metrics_rows.append({
            "model": name,
            "split": "val",
            **model_metrics,
        })

    # 7. Learning-rate experiment.
    learning_rate_rows = []

    for learning_rate in [0.01, 0.1, 1.0]:
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=learning_rate,
            random_state=42,
        )

        gb_model.fit(X_train, y_train)

        gb_prediction = gb_model.predict(X_val)

        gb_metrics = calculate_metrics(
            y_val,
            gb_prediction,
        )

        learning_rate_rows.append({
            "learning_rate": learning_rate,
            "val_rmse": gb_metrics["RMSE"],
            "val_r2": gb_metrics["R2"],
        })

    learning_rate_df = pd.DataFrame(learning_rate_rows)

    learning_rate_df.to_csv(
        OUTPUT_DIR / "part_b_gb_learning_rate.csv",
        index=False,
        float_format="%.4f",
    )

    # 8. Rank models by validation RMSE.
    ranked_models = sorted(
        validation_rmse.items(),
        key=lambda item: item[1],
    )

    best_model_name = ranked_models[0][0]
    final_model = models[best_model_name]

    # 9. Cross-validation for the two best validation models.
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_rows = []

    for model_name, _ in ranked_models[:2]:
        cv_model = models[model_name]

        cv_scores = cross_val_score(
            cv_model,
            X_train_val,
            y_train_val,
            cv=kfold,
        )

        cv_rows.append({
            "model": model_name,
            "mean_r2": cv_scores.mean(),
            "std_r2": cv_scores.std(),
        })

    cv_df = pd.DataFrame(cv_rows)

    cv_df.to_csv(
        OUTPUT_DIR / "part_b_cv.csv",
        index=False,
        float_format="%.4f",
    )

    # 10. Evaluate the selected final model on test data exactly once.
    test_prediction = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
    )

    metrics_rows.append({
        "model": best_model_name,
        "split": "test",
        **test_metrics,
    })

    # 11. Save all model metrics.
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "part_b_metrics.csv",
        index=False,
        float_format="%.4f",
    )

    # 12. Print results for validation.
    print("Part B completed successfully.")

    print("\nSplit sizes:")
    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    print("\nScaling checks:")
    print(f"Training scaled max absolute mean: {train_scaled_max_abs_mean:.12f}")
    print(f"Test scaled max absolute mean: {test_scaled_max_abs_mean:.12f}")

    print("\nBest decision tree depth:", best_depth)

    print("\nValidation model ranking:")
    for rank, (name, score) in enumerate(ranked_models, start=1):
        print(f"{rank}. {name}: RMSE = {score:.4f}")

    print("\nFinal selected model:", best_model_name)
    print("Final test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPart B output files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
