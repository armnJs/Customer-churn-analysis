import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Churn Analysis & Machine Learning Prediction\n",
    "## Complete End-to-End Data Science Project\n",
    "\n",
    "### Project Overview\n",
    "Customer churn is one of the most critical metrics for subscription and SaaS businesses. Retaining an existing customer is significantly cheaper than acquiring a new one. This project builds an end-to-end Machine Learning pipeline to predict customer churn, identify key risk drivers (e.g., payment delays, high support calls, contract types), and evaluate models using business-critical metrics.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, roc_curve, confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "# Visual Styling\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams['font.sans-serif'] = 'Arial'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Load Dataset & Inspect Schema"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('customer.csv')\n",
    "print(\"Original Dataset Shape:\", df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Data Quality Audit & Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"--- Missing Values Per Column ---\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Handle missing data by dropping incomplete records\n",
    "df_clean = df.dropna().copy()\n",
    "\n",
    "# Ensure numeric casting\n",
    "num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']\n",
    "for col in num_cols:\n",
    "    df_clean[col] = pd.to_numeric(df_clean[col])\n",
    "\n",
    "df_clean['Churn'] = pd.to_numeric(df_clean['Churn']).astype(int)\n",
    "print(\"\\nCleaned Dataset Shape:\", df_clean.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Exploratory Data Analysis (EDA)\n",
    "Visualizing patterns between customer behavior and churn status."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "\n",
    "# 1. Churn Rate Distribution\n",
    "churn_counts = df_clean['Churn'].value_counts()\n",
    "axes[0, 0].pie(churn_counts, labels=['Retained (0)', 'Churned (1)'], autopct='%1.1f%%',\n",
    "               startangle=140, colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0), wedgeprops=dict(width=0.4))\n",
    "axes[0, 0].set_title('Overall Churn Rate Distribution', fontsize=12, fontweight='bold')\n",
    "\n",
    "# 2. Contract Length vs Churn\n",
    "contract_churn = df_clean.groupby('Contract Length')['Churn'].mean().reset_index()\n",
    "sns.barplot(data=contract_churn, x='Contract Length', y='Churn', hue='Contract Length', palette='Blues_d', ax=axes[0, 1], legend=False)\n",
    "axes[0, 1].set_title('Churn Rate by Contract Type', fontsize=12, fontweight='bold')\n",
    "axes[0, 1].set_ylabel('Churn Rate')\n",
    "\n",
    "# 3. Support Calls vs Churn\n",
    "sns.boxplot(data=df_clean, x='Churn', y='Support Calls', hue='Churn', palette=['#2ecc71', '#e74c3c'], ax=axes[1, 0], legend=False)\n",
    "axes[1, 0].set_xticklabels(['Retained', 'Churned'])\n",
    "axes[1, 0].set_title('Support Calls Volume by Status', fontsize=12, fontweight='bold')\n",
    "\n",
    "# 4. Payment Delay Distribution\n",
    "sns.kdeplot(data=df_clean, x='Payment Delay', hue='Churn', common_norm=False, palette=['#2ecc71', '#e74c3c'], fill=True, alpha=0.4, ax=axes[1, 1])\n",
    "axes[1, 1].set_title('Payment Delay Density Distribution', fontsize=12, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Feature Encoding & Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_encoded = df_clean.drop(columns=['CustomerID']).copy()\n",
    "\n",
    "# Categorical Encoding using independent LabelEncoders\n",
    "encoders = {}\n",
    "cat_cols = ['Gender', 'Subscription Type', 'Contract Length']\n",
    "for col in cat_cols:\n",
    "    le = LabelEncoder()\n",
    "    df_encoded[col] = le.fit_transform(df_encoded[col])\n",
    "    encoders[col] = le\n",
    "\n",
    "# Correlation Heatmap\n",
    "plt.figure(figsize=(10, 7))\n",
    "sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)\n",
    "plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Train-Test Split & Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_encoded.drop(columns=['Churn'])\n",
    "y = df_encoded['Churn']\n",
    "\n",
    "# Stratified 70/30 Train/Test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.30, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Feature scaling for linear models\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Training set shape: {X_train.shape}\")\n",
    "print(f\"Testing set shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Model Training & Evaluation Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=4, class_weight='balanced', random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42),\n",
    "    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "for name, model in models.items():\n",
    "    if name == 'Logistic Regression':\n",
    "        model.fit(X_train_scaled, y_train)\n",
    "        y_pred = model.predict(X_test_scaled)\n",
    "        y_prob = model.predict_proba(X_test_scaled)[:, 1]\n",
    "    else:\n",
    "        model.fit(X_train, y_train)\n",
    "        y_pred = model.predict(X_test)\n",
    "        y_prob = model.predict_proba(X_test)[:, 1]\n",
    "    \n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec = precision_score(y_test, y_pred)\n",
    "    rec = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    auc = roc_auc_score(y_test, y_prob)\n",
    "    \n",
    "    results[name] = {\n",
    "        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc,\n",
    "        'y_pred': y_pred, 'y_prob': y_prob\n",
    "    }\n",
    "    \n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {auc:.2f})\", linewidth=2)\n",
    "\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.50)')\n",
    "plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.legend(loc='lower right')\n",
    "plt.show()\n",
    "\n",
    "# Display Performance Table\n",
    "results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]\n",
    "print(\"\\n--- Model Performance Comparison Table ---\")\n",
    "print(results_df.round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 8: Confusion Matrix Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n",
    "\n",
    "for idx, (name, res) in enumerate(results.items()):\n",
    "    cm = confusion_matrix(y_test, res['y_pred'])\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])\n",
    "    axes[idx].set_title(f'Confusion Matrix: {name}', fontsize=12, fontweight='bold')\n",
    "    axes[idx].set_ylabel('Actual Label')\n",
    "    axes[idx].set_xlabel('Predicted Label')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 9: Feature Importance & Business Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_model = models['Random Forest']\n",
    "importance_df = pd.DataFrame({\n",
    "    'Feature': X.columns,\n",
    "    'Importance': rf_model.feature_importances_\n",
    "}).sort_values(by='Importance', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(9, 5))\n",
    "sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)\n",
    "plt.title('Random Forest - Key Predictive Churn Drivers', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Gini Importance')\n",
    "plt.show()\n",
    "\n",
    "print(\"Top 5 Critical Predictors:\")\n",
    "print(importance_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 10: Decision Tree Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(20, 10))\n",
    "plot_tree(\n",
    "    models['Decision Tree'],\n",
    "    feature_names=X.columns,\n",
    "    class_names=['Retained', 'Churned'],\n",
    "    filled=True,\n",
    "    rounded=True,\n",
    "    fontsize=9\n",
    ")\n",
    "plt.title('Decision Tree Rules Visualization (Max Depth = 4)', fontsize=16, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 11: Single Customer Inference Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_customer_churn(sample_dict):\n",
    "    sample_df = pd.DataFrame([sample_dict])\n",
    "    for col in cat_cols:\n",
    "        sample_df[col] = encoders[col].transform(sample_df[col])\n",
    "    prob = models['Random Forest'].predict_proba(sample_df)[0][1]\n",
    "    status = \"HIGH CHURN RISK\" if prob >= 0.50 else \"LOW CHURN RISK\"\n",
    "    return status, round(prob * 100, 2)\n",
    "\n",
    "# Test Sample Customer\n",
    "test_customer = {\n",
    "    'Age': 45,\n",
    "    'Gender': 'Female',\n",
    "    'Tenure': 5,\n",
    "    'Usage Frequency': 4,\n",
    "    'Support Calls': 6,\n",
    "    'Payment Delay': 18,\n",
    "    'Subscription Type': 'Basic',\n",
    "    'Contract Length': 'Monthly',\n",
    "    'Total Spend': 350.0,\n",
    "    'Last Interaction': 22\n",
    "}\n",
    "\n",
    "risk_status, risk_pct = predict_customer_churn(test_customer)\n",
    "print(f\"Inference Result: {risk_status} ({risk_pct}% probability of churn)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

file_path = r"d:\Armaan\Data visualization\Customer churn analysis\Customer Churn analysis.ipynb"
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print("Successfully written notebook structure JSON.")
