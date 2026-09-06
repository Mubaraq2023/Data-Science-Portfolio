# Mobile Money Fraud Detection

A machine learning pipeline for detecting fraudulent mobile-money transactions from transaction-level data.

## Overview

Mobile-money platforms process high volumes of transactions in which fraud is rare but costly, making it a classic class-imbalanced classification problem. This project builds an end-to-end workflow, from raw transaction data to a trained, evaluated, and serialized model that flags transactions as fraudulent or legitimate based on behavioral, temporal, and merchant/channel-related signals.

The notebook walks through the full data science lifecycle:

1. Data loading and cleaning
2. Exploratory data analysis (EDA)
3. Feature preprocessing and model development
4. Model evaluation, tuning, and selection

## Key Features

- **Data cleaning**: handles missing values, duplicate detection, and datatype correction (datetime parsing, categorical encoding, memory-efficient integer types).
- **Exploratory analysis**: visualizes class imbalance, transaction amount distribution, monthly fraud trends, and fraud rates broken down by channel, merchant category, location, age group, weekday/weekend, and bank.
- **Feature engineering awareness**: works with a rich, pre-engineered feature set including rolling transaction statistics (24h/7d/total), velocity and risk scores, and cyclical time encodings.
- **Model comparison**: trains and evaluates four classifiers — Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting, using class-weight balancing to address fraud rarity.
- **Hyperparameter tuning**: uses `RandomizedSearchCV` to tune the HistGradientBoosting and Random Forest models.
- **Evaluation suite**: reports confusion matrices, PR-AUC, ROC-AUC, precision, recall, and F1-score for each model, and selects the best-performing model automatically.
- **Model persistence**: serializes the final selected model to disk for downstream use.

## Tech Stack / Dependencies

- **Language**: Python 3.13
- **Environment**: Jupyter Notebook
- **Core libraries**:
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization
  - `scikit-learn` — preprocessing, modeling, evaluation, and hyperparameter search
  - `joblib` — model serialization

## Installation / Setup

1. **Clone the repository** (or copy the notebook into your project directory).

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
   ```

4. **Add the dataset**: place `nibss_fraud_dataset.csv` in the same directory as the notebook. The dataset is expected to include a `transaction_id` index column along with transaction, customer, and behavioral features (see [Project Structure](#project-structure) below).

5. **Launch Jupyter**:

   ```bash
   jupyter notebook Mobile-Money_Fraud_Detection.ipynb
   ```

## Usage

Run the notebook cells sequentially from top to bottom. The main stages are:

**1. Load and inspect the data**

```python
df = pd.read_csv("nibss_fraud_dataset.csv", low_memory=False, index_col="transaction_id")
```

**2. Preprocess features**

```python
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
], sparse_threshold=0.3)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
```

**3. Train and compare models**

```python
models = {
    'Logistic Regression': Pipeline([('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
    'Decision Tree': Pipeline([('model', DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42))]),
    'Random Forest': Pipeline([('model', RandomForestClassifier(class_weight='balanced', n_estimators=200, n_jobs=-1, random_state=42))]),
    'HistGradientBoosting': Pipeline([('model', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))])
}
```

**4. Persist the final model**

```python
import joblib
joblib.dump(final_model, "fraud_detection_model.pkl")
```

To reuse the trained model elsewhere:

```python
import joblib
model = joblib.load("fraud_detection_model.pkl")
predictions = model.predict(new_data)
```

## Configuration

This project has no environment variables or external configuration files. The only required input is the dataset file, `nibss_fraud_dataset.csv`, which must be present in the working directory before running the notebook.

## Project Structure

The notebook expects a dataset with the following (approximate) schema:

| Category | Columns |
|---|---|
| Identifiers | `transaction_id`, `customer_id` |
| Transaction details | `amount`, `channel`, `merchant_category`, `bank`, `location`, `age_group`, `timestamp` |
| Time features | `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_hour` |
| Behavioral aggregates | `tx_count_24h`, `amount_sum_24h`, `amount_mean_7d`, `amount_std_7d`, `tx_count_total`, `amount_mean_total`, `amount_std_total` |
| Derived risk signals | `channel_diversity`, `amount_vs_mean_ratio`, `online_channel_ratio`, `velocity_score`, `merchant_risk_score` |
| Target | `is_fraud`, `fraud_technique` |

**Outputs:**

- `fraud_detection_model.pkl` — the serialized final model, saved via `joblib` after training and selection.

## Contributing

Contributions are welcome. If you'd like to improve the analysis, add new models, or extend the feature set:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes with clear messages.
4. Open a pull request describing the change and its motivation.

Please keep notebook cells well-documented and avoid committing large data files or trained model artifacts unless necessary.


