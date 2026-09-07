# California House Price Prediction

A machine learning notebook that predicts median house values in California districts from census-derived housing and demographic data.

## Overview

This project builds an end-to-end regression pipeline to estimate `median_house_value` for California housing blocks based on features such as location, income, and housing characteristics. It walks through the full data science workflow - exploratory data analysis, data cleaning, preprocessing, model comparison, hyperparameter tuning, and model interpretation — culminating in a serialized model ready for inference on new data.

The project exists as a practical, reproducible example of applying classical regression techniques (linear, regularized, and ensemble-based) to a real-world tabular dataset, with attention to common pitfalls such as capped target values and skewed features.

## Key Features

- **Exploratory Data Analysis (EDA)**: distribution plots, outlier detection (boxplots), correlation analysis, and geographic visualization of price patterns across California.
- **Data cleaning**: removal of artificially capped target values (houses valued at exactly $500,000) and log-transformation of heavily right-skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`).
- **Preprocessing pipeline**: `ColumnTransformer`-based pipeline combining median imputation and standard scaling for numerical features with one-hot encoding for categorical features (`ocean_proximity`).
- **Model comparison**: benchmarks seven regression approaches — Linear, Ridge, Lasso, Polynomial (Ridge), Support Vector Regression, Random Forest, and Gradient Boosting.
- **Hyperparameter tuning**: `GridSearchCV` (5-fold cross-validation) for Ridge, Random Forest, and Gradient Boosting models.
- **Model evaluation and diagnostics**: MAE, MSE, RMSE, and R² metrics, plus actual-vs-predicted and residual plots to assess fit and heteroscedasticity.
- **Feature importance / interpretability**: feature importances for tree-based models or coefficients for linear models, depending on which model performs best.
- **Model persistence**: the final trained model is serialized with `joblib` for reuse without retraining.

## Tech Stack / Dependencies

- **Language**: Python 3
- **Environment**: Jupyter Notebook
- **Core libraries**:
  - `numpy`, `pandas` - Data manipulation
  - `matplotlib`, `seaborn` - Visualization
  - `scikit-learn` — Preprocessing, Pipelines, Models, and Evaluation (`train_test_split`, `GridSearchCV`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `PolynomialFeatures`, `SimpleImputer`, `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, and regression metrics)
  - `joblib` - Model serialization

## Installation / Setup

1. **Clone or download the project**, ensuring the notebook (`Kaggle_California_House_Prediction.ipynb`) and the dataset (`california_housing.csv`) are in the same directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Kaggle_California_House_Prediction.ipynb
   ```

## Dataset

The notebook expects a file named `california_housing.csv` in the working directory, containing the classic California housing census features (`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`) and the target column `median_house_value`.

> 📎 **Dataset link:** _[https://www.kaggle.com/datasets/camnugent/california-housing-prices/data]_

## Usage

Run the notebook cells sequentially. The core stages are:

**1. Load the data**
```python
df = pd.read_csv("california_housing.csv")
```

**2. Clean and transform**
```python
# Remove artificially capped target values
df = df[df["median_house_value"] < 500000].reset_index(drop=True)

# Log-transform skewed features
cols_to_transform = ["total_rooms", "total_bedrooms", "population", "households"]
for col in cols_to_transform:
    df[col] = np.log1p(df[col])
```

**3. Build the preprocessing pipeline**
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ]
)
```

**4. Train and compare models, then tune the best candidates** using `GridSearchCV`.

**5. Evaluate and select the final model** based on R² score, then persist it:
```python
import joblib
joblib.dump(final_model, "house_price_model.pkl")
```

**6. Load the model and predict on new data**
```python
loaded_model = joblib.load("house_price_model.pkl")
prediction = loaded_model.predict(new_data)
```

## Configuration

No environment variables are required. The only configurable parameters live inline in the notebook:

- `test_size` / `random_state` in the train/test split (default: `0.2` / `42`)
- Hyperparameter grids for Ridge, Random Forest, and Gradient Boosting (`rf_params`, `gb_params`, `ridge_params`)
- The capped-value threshold used for filtering the target (`500000`)

## Project Structure

```
.
├── Kaggle_California_House_Prediction.ipynb   # Main analysis and modeling notebook
├── california_housing.csv                      # Input dataset (not included — see Dataset section)
└── house_price_model.pkl                       # Serialized final model (generated after running the notebook)
```

## Results Summary

- The strongest single predictor of house value is `median_income` (correlation ≈ 0.69; ~47% feature importance in tree-based models).
- Tree-based ensembles (Random Forest, Gradient Boosting) substantially outperform linear models, capturing non-linear relationships and geographic effects.
- After hyperparameter tuning, the best-performing model achieves an R² of roughly 0.80 on the held-out test set, with prediction error increasing at higher price ranges (heteroscedasticity).

## Contributing

This is currently a single-notebook data science exercise rather than a packaged open-source library. If you'd like to extend it (e.g., additional feature engineering, alternative models, or a deployment script), feel free to fork the notebook and submit improvements via pull request. Please keep new analysis steps documented with markdown cells, consistent with the existing notebook style.
