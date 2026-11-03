# Delhivery Last-Mile Delay Intelligence Pipeline

## Project Overview

This project builds an end-to-end machine learning pipeline to analyse and predict last-mile shipment delays. It covers data cleaning, exploratory analysis, delivery-time regression, delay classification, hub segmentation, model packaging, and predictions for new shipments.

The objective is to help operations teams identify shipments at risk of delay and prioritise hubs that may require operational intervention.

## Repository Structure

```text
delhivery-last-mile-delay-intelligence/
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   └── shipment_prep.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── models/
└── outputs/
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/bharath2819/delhivery-last-mile-delay-intelligence.git
cd delhivery-last-mile-delay-intelligence
```

### 2. Create and activate a virtual environment

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the complete pipeline

Run the following scripts from the project root in the given order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The scripts generate the cleaned modelling dataset, evaluation results, hub segments, packaged model, and new-shipment action recommendations.

Dependencies are pinned in `requirements.txt`. The packaged pickle should be loaded using a compatible scikit-learn environment. If the environment differs, rerun `package_model.py` to regenerate the model.

## Data Findings

The historical shipment dataset initially contained 1,224 rows. After removing 24 duplicate records, 1,200 unique shipments remained.

The data preparation process included:

- Removing exact duplicate records.
- Standardising service type names.
- Handling missing payment modes.
- Converting declared values into numeric values after removing currency symbols and commas.
- Treating parcel weights above 100 kg as invalid sentinel values.
- Imputing missing distance, parcel weight, and declared value using historical medians.
- Merging hub information with shipment records.
- Creating modelling features without using the delivery-time target as an input.

### Imputation Values

| Field | Historical Median |
|---|---:|
| Distance | 314.95 km |
| Parcel weight | 1.83 kg |
| Declared value | ₹1,121.50 |

### Key Findings

- The final modelling dataset contains 1,200 shipments.
- 278 shipments were delayed, resulting in an overall delay rate of 23.17%.
- The correlation between delivery hours and promised hours was 0.8372.
- The correlation between delivery hours and distance was 0.4013.
- The correlation between delay status and distance was 0.5455.
- One delivery-time observation was identified as an IQR outlier and retained rather than automatically removed.

The findings indicate that distance and promised delivery time are useful factors for understanding shipment delays. However, correlation does not establish causation.

Generated outputs include the modelling table, data quality checks, correlation matrix, hub profile, and delay breakdowns by service type, distance band, and hub tier.

## Model Results

### Regression

The regression model predicts delivery time in hours.

The historical dataset was divided into training, validation, and test sets.

| Dataset | Rows |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Test | 180 |

Feature scaling was fitted using training data only and then applied to the validation and test sets.

#### Validation Model Comparison

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Mean baseline | 18.8552 | -0.0212 |
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

Linear Regression achieved the lowest validation RMSE and was selected as the final regression model.

#### Model Experiments

The Decision Tree depth experiment demonstrated a bias-variance trade-off. A shallow tree underfitted the data, while an unrestricted tree showed signs of overfitting. Depth 6 achieved the best validation RMSE of 6.2796 hours.

For Gradient Boosting, a learning rate of 0.1 performed best among the tested values, achieving a validation R² of 0.9306.

Five-fold cross-validation on the combined training and validation data produced the following results:

- Linear Regression: mean R² of 0.9261, standard deviation of 0.0048.
- Gradient Boosting: mean R² of 0.9353, standard deviation of 0.0045.

Although Gradient Boosting achieved a slightly higher cross-validation R², Linear Regression was retained because it had the lowest validation RMSE, following the model-selection rule.

#### Final Test Performance

| Metric | Test Result |
|---|---:|
| MAE | 3.5671 hours |
| MSE | 20.4624 |
| RMSE | 4.5235 hours |
| R² | 0.9493 |

The test RMSE indicates an error scale of approximately 4.52 hours. This is an evaluation result on the held-out test set and is not a guarantee of future prediction accuracy.

### Classification

The classification model predicts whether a shipment will be delayed.

The target is a binary delay label. Stratified splitting was used to preserve the proportion of delayed and on-time shipments across the training, validation, and test sets.

Each dataset contained:

| Dataset | Rows |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Test | 180 |

#### Model Selection

The experiments included an all-on-time baseline, Logistic Regression, Decision Tree, and Random Forest.

Feature selection experiments compared 5, 7, 10, 12, and 17 features. The seven-feature configuration performed best among the tested feature-count options.

The selected features were:

1. Distance
2. Hub tier
3. Hub daily capacity
4. Number of stops
5. Promised delivery hours
6. Rain indicator
7. Sale-event indicator

The final configuration used a Random Forest with:

- 100 estimators
- Unlimited maximum depth
- Minimum leaf size of 1
- Random state of 42

The classification threshold was selected using validation recall, with a minimum target of 0.80. A threshold of 0.50 was selected, achieving validation recall of 0.8333.

#### Final Test Performance

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 score | 0.7209 |
| True negatives | 125 |
| False positives | 14 |
| False negatives | 10 |
| True positives | 31 |

In practical terms:

- Precision of 0.6889 means roughly 7 out of 10 flagged shipments were actually delayed.
- Recall of 0.7561 means the model identified roughly 8 out of 10 delayed shipments in the test set.
- The model missed 10 delayed shipments and generated 14 false alerts in the test set.

The model can support early intervention, but its predictions should not be treated as certain outcomes.

## Hub Segmentation and Operational Analysis

Hub segmentation groups hubs according to observed delivery performance instead of relying only on their assigned tiers.

The hub profile was created using four features:

- Average delivery time
- Delay rate
- Average delivery distance
- Average number of stops

These features were scaled before applying K-Means clustering.

### Choosing the Number of Clusters

K-Means was evaluated for K values from 2 to 5.

| K | Across-to-within Distance Ratio | Cluster Sizes |
|---|---:|---|
| 2 | 2.8940 | 15, 9 |
| 3 | 5.3928 | 9, 8, 7 |
| 4 | 6.3047 | 8, 7, 5, 4 |
| 5 | 6.8866 | 8, 7, 4, 3, 2 |

Although K=4 and K=5 produced higher separation ratios, they also created smaller clusters. K=3 was selected because it provided useful separation while keeping every cluster at seven or more hubs, making the groups more practical for operational planning.

The three segments were named according to their average hub-level delay rates.

### Segment Profile

| Segment | Hubs | Mean Delivery Time | Delay Rate | Mean Distance | Mean Stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 hrs | 2.57% | 188.93 km | 55.37 |
| Watch | 8 | 56.8137 hrs | 15.50% | 314.18 km | 70.72 |
| Escalate | 9 | 64.6504 hrs | 46.00% | 470.48 km | 84.34 |

### Escalate Hubs

The following nine hubs belong to the Escalate segment:

| Hub | City | Hub Tier |
|---|---|---:|
| HUB13 | Nagpur | 2 |
| HUB15 | Kochi | 2 |
| HUB17 | Patna | 3 |
| HUB18 | Ranchi | 3 |
| HUB19 | Guwahati | 3 |
| HUB20 | Raipur | 3 |
| HUB21 | Bhubaneswar | 3 |
| HUB23 | Siliguri | 3 |
| HUB24 | Agartala | 3 |

These hubs should receive closer operational monitoring because their segment has the highest observed delay rate.

### Tier-versus-Segment Mismatches

Four hubs do not match the expected segment associated with their assigned tier:

| Hub | City | Tier | Actual Segment |
|---|---|---:|---|
| HUB07 | Kolkata | 1 | Watch |
| HUB13 | Nagpur | 2 | Escalate |
| HUB15 | Kochi | 2 | Escalate |
| HUB22 | Dehradun | 3 | Watch |

These mismatches are investigation priorities. They do not automatically mean that a hub's assigned tier should change. Tier represents a broad hub category, while segmentation reflects observed shipment performance.

## Model Packaging and New-Shipment Predictions

`package_model.py` rebuilds the selected classification pipeline, fits it using the designated training split, and saves the model and supporting configuration files.

The packaged artefacts include:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

The model reload check scored 180 test rows and recorded a maximum absolute probability difference of 0.000000 after reloading the model.

`predict_delay.py` loads the packaged model and applies it to the new shipment dataset. It uses the stored imputation values, feature order, classification threshold, and hub segments.

### New-Shipment Action Summary

The new shipment dataset contains 48 shipments.

| Recommended Action | Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |
| **Total** | **48** |

The action logic is:

- A flagged shipment from an Escalate hub is sent for hub-manager escalation.
- A flagged shipment from another segment receives a proactive customer call.
- A shipment below the selected threshold receives no action.

The output is saved to `outputs/new_shipment_actions.csv`.

These are model-generated recommendations, not confirmed delivery outcomes.

## Operations Recommendation

The operations team should use the classification output as an early-warning queue rather than as an automatic decision system.

1. **Prioritise the Escalate segment.** Review the nine hubs in this group, focusing first on hubs with high observed delay rates.
2. **Investigate tier mismatches.** Review HUB07, HUB13, HUB15, and HUB22 to understand why their observed performance differs from the expected tier-based segment.
3. **Act on flagged shipments.** Escalate flagged shipments from Escalate hubs to hub managers and arrange proactive customer calls for other flagged shipments.
4. **Review false alerts and missed delays.** The test results include 14 false positives and 10 false negatives. Operations feedback can help identify where the model needs improvement.
5. **Track segment performance.** Recalculate hub-level delivery time and delay rate regularly to identify hubs whose performance improves or deteriorates.

The new-shipment recommendations identify six hub-manager escalations and two proactive customer calls among 48 shipments. These counts describe the current prediction run and should not be interpreted as future delay rates.

## Production Plan

The project contains several components that support a future production deployment.

| Readiness Area | Evidence in This Project |
|---|---|
| Repeatable data preparation | Shared cleaning and feature logic in `src/shipment_prep.py` |
| Reproducible dependencies | Pinned versions in `requirements.txt` |
| Model artefact | Packaged classification pipeline in `models/delay_model.pkl` |
| Stored inference configuration | Threshold, feature order, and imputation values saved under `models/` |
| Inference workflow | `predict_delay.py` generates shipment-level predictions and actions |
| Validation checks | Data quality, model evaluation, and reload checks saved under `outputs/` |

These components provide a foundation for deployment. They do not mean the pipeline is already running as a monitored production service.

### Monitoring Metrics

The following are proposed operational triggers. They should be reviewed and calibrated after collecting real production data.

| Metric | Suggested Trigger | Response |
|---|---|---|
| Missing or invalid prediction inputs | More than 1% of daily shipments | Investigate upstream data ingestion and validation |
| Delayed-shipment recall | Below 0.75 over a rolling labelled evaluation window | Review threshold, features, and model performance |
| False-alert rate among flagged shipments | Above 35% over a rolling labelled evaluation window | Review alert quality and operational impact |

Monitoring should also track prediction volume, feature distributions, hub-level delay rates, and the number of manager escalations.

### Retraining Plan

A practical starting point is to evaluate model performance monthly and consider retraining quarterly, provided enough newly labelled shipment data is available.

Retraining should also be considered when:

- Delay recall falls below the agreed operational target.
- Input feature distributions change substantially.
- New hubs, service types, or operational processes are introduced.
- Changes in routing, capacity, or dispatch practices affect delivery patterns.

For sale-event monitoring, compare the period from 15–28 September with the same sale-event window in another comparable cycle. Avoid comparing it only with ordinary non-sale weeks, since shipment volume and operating conditions may differ.

Examples of potential drift include longer average distances, changes in the number of stops, increased sale-event volume, changes in service mix, or a shift in the proportion of shipments handled by each hub tier.

## Limitations

1. **Synthetic or limited historical data:** Model performance on this dataset may not represent real-world performance across all Delhivery operations.
2. **Limited feature coverage:** The current features do not capture every operational factor, such as live traffic, weather severity, route disruptions, staffing levels, or real-time vehicle availability.
3. **Historical relationships can change:** Routing changes, new hubs, seasonal demand, and operational improvements can affect model performance.
4. **Classification errors remain:** The test set contains both false alerts and missed delayed shipments. Human review is important before operational action.
5. **Clustering is descriptive:** Hub segments summarise observed patterns. They do not prove why a hub is delayed or establish that its assigned tier is incorrect.
6. **Model portability:** Pickle artefacts can depend on the scikit-learn version and environment used to create them. Use the pinned dependencies or regenerate the model with `package_model.py`.

## Conclusion

This project demonstrates an end-to-end last-mile delay intelligence workflow, from shipment data preparation to model evaluation, hub segmentation, and new-shipment recommendations.

The regression model estimates delivery time, the classification model identifies shipments at risk of delay, and the segmentation analysis helps operations teams prioritise hubs. Together, these outputs provide a starting point for more proactive shipment monitoring and operational decision-making.