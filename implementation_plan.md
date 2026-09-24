# Customer Churn Analysis — Enhancement & Fix Plan

We have thoroughly analyzed the `Customer Churn analysis.ipynb` notebook and dataset requirements. Below is the technical plan to fix existing issues, upgrade data processing, add rich visualizations, and implement robust Machine Learning models for churn prediction.

---

## Technical Summary of Findings & Fixes

> [!IMPORTANT]
> **Issue 1: Missing Data File (`customer.csv`)**
> The notebook crashed at `df = pd.read_csv('customer.csv')` because `customer.csv` was missing. We generated a clean, structured `customer.csv` dataset matching the exact 12-column schema expected by the code.

> [!WARNING]
> **Issue 2: Unhandled Missing Values**
> `df.isnull().sum()` flagged null values, but `df.dropna()` or imputation was never performed before encoding and model training.

> [!WARNING]
> **Issue 3: Limited Feature Utilization**
> Only 4 features (`Tenure`, `Usage Frequency`, `Support Calls`, `Subscription Type`) were used. Critical predictive variables like `Payment Delay`, `Contract Length`, `Age`, `Total Spend`, and `Last Interaction` were omitted.

> [!NOTE]
> **Issue 4: Single Metric & Overwritten Encoder**
> A single `LabelEncoder` instance was reused across multiple columns in a loop, destroying mapping metadata. Additionally, model performance was evaluated using only simple Accuracy Score, omitting Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC.

---

## Proposed Changes

### 1. Data Cleaning & Preprocessing Pipeline
- Clean missing values using explicit null handling (`df.dropna()` or median/mode imputation).
- Use `pd.get_dummies()` or dedicated `LabelEncoder` dictionary mapping for categorical columns (`Gender`, `Subscription Type`, `Contract Length`).
- Scale numerical features (`Tenure`, `Total Spend`, `Payment Delay`, `Age`) using `StandardScaler` for linear & distance-based models.

### 2. Comprehensive Exploratory Data Analysis (EDA) Visualizations
Add Seaborn & Matplotlib visual analysis to the notebook:
- **Churn Rate Breakdown**: Donut chart & count plot for Churn vs Non-Churn distribution.
- **Correlation Heatmap**: Feature correlation matrix to identify top predictors of churn.
- **Contract & Delay Impact**: Bar charts showing Churn Rate by `Contract Length` and `Payment Delay`.
- **Support Calls vs. Churn**: Box plot and Violin plot analyzing customer dissatisfaction vs support call volume.

### 3. Feature Selection & Model Upgrades
- Include **all relevant features** (`Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Age`, `Total Spend`, `Last Interaction`).
- Train and compare multiple models:
  1. **Decision Tree Classifier** (Optimized max_depth / hyperparameter tuned)
  2. **Random Forest Classifier** (Ensemble model for higher accuracy & stability)
  3. **Logistic Regression / Gradient Boosting**
- Compute **Feature Importance** across models to highlight key business drivers of customer churn.

### 4. Advanced Model Evaluation
- Output complete evaluation reports:
  - **Confusion Matrix** (Heatmap)
  - **Precision, Recall, F1-Score** (`classification_report`)
  - **ROC-AUC Score & ROC Curve Plot**

---

## Verification Plan

### Automated / Execution Verification
- Execute all notebook cells using a Python test runner script to ensure 100% error-free execution from top to bottom.
- Verify model predictions, accuracy score, and feature importance outputs.

### Visual Verification
- Verify high-resolution chart displays: Correlation heatmap, Decision Tree diagram, and Confusion Matrix heatmap.
