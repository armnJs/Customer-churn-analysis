import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'

# Create figures directory
fig_dir = r"d:\Armaan\Data visualization\Customer churn analysis\images"
os.makedirs(fig_dir, exist_ok=True)

print("1. Loading Dataset...")
df = pd.read_csv(r"d:\Armaan\Data visualization\Customer churn analysis\customer.csv")
print(f"Dataset shape: {df.shape}")

# Drop missing rows / clean nulls
print("\n2. Data Cleaning...")
df_clean = df.dropna().copy()
print(f"Shape after dropping nulls: {df_clean.shape}")

# Ensure numeric types
num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col])

df_clean['Churn'] = pd.to_numeric(df_clean['Churn']).astype(int)

# --- EDA VISUALIZATIONS ---
print("\n3. Generating EDA Visualizations...")

# Visual 1: Churn Distribution (Donut Chart)
plt.figure(figsize=(6, 6))
churn_counts = df_clean['Churn'].value_counts()
plt.pie(churn_counts, labels=['Retained (0)', 'Churned (1)'], autopct='%1.1f%%',
        startangle=140, colors=['#2ecc71', '#e74c3c'], explode=(0.05, 0), wedgeprops=dict(width=0.4))
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'churn_distribution.png'), dpi=300)
plt.close()

# Visual 2: Contract Length vs Churn Rate
plt.figure(figsize=(8, 5))
contract_churn = df_clean.groupby('Contract Length')['Churn'].mean().reset_index()
sns.barplot(data=contract_churn, x='Contract Length', y='Churn', hue='Contract Length', palette='Blues_d', legend=False)
plt.title('Churn Rate by Contract Length', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate', fontsize=12)
plt.xlabel('Contract Length', fontsize=12)
for index, row in contract_churn.iterrows():
    plt.text(index, row['Churn'] + 0.01, f"{row['Churn']*100:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'churn_by_contract.png'), dpi=300)
plt.close()

# Visual 3: Support Calls vs Churn Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x='Churn', y='Support Calls', hue='Churn', palette=['#2ecc71', '#e74c3c'], legend=False)
plt.xticks([0, 1], ['Retained', 'Churned'])
plt.title('Support Calls Volume by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Customer Status', fontsize=12)
plt.ylabel('Number of Support Calls', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'support_calls_churn.png'), dpi=300)
plt.close()

# Visual 4: Payment Delay Distribution
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_clean, x='Payment Delay', hue='Churn', common_norm=False, palette=['#2ecc71', '#e74c3c'], fill=True, alpha=0.4)
plt.title('Payment Delay Density Distribution by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Payment Delay (Days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'payment_delay_density.png'), dpi=300)
plt.close()

# --- PREPROCESSING & ENCODING ---
print("\n4. Encoding & Feature Selection...")
df_encoded = df_clean.copy()
df_encoded.drop(columns=['CustomerID'], inplace=True)

encoders = {}
cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# Visual 5: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_encoded.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'correlation_heatmap.png'), dpi=300)
plt.close()

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL TRAINING & EVALUATION WITH CLASS WEIGHT BALANCING ---
print("\n5. Training Machine Learning Models (Class Weighted)...")

models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=4, class_weight='balanced', random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42),
    'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', random_state=42)
}

results = {}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc,
        'y_pred': y_pred, 'y_prob': y_prob
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'roc_curves.png'), dpi=300)
plt.close()

# Print Performance Table
results_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
print("\n--- Model Comparison Summary ---")
print(results_df.round(4))

# Visual 6: Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(
    models['Decision Tree'],
    feature_names=X.columns,
    class_names=['No Churn', 'Churn'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Decision Tree Visual Boundary (Max Depth = 4)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'decision_tree_structure.png'), dpi=300)
plt.close()

# Visual 7: Random Forest Feature Importance
rf_model = models['Random Forest (Balanced)']
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=importances, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Random Forest - Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.xlabel('Gini Importance', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=300)
plt.close()

# Visual 8: Confusion Matrix for Random Forest
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, results['Random Forest (Balanced)']['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Random Forest (Balanced)', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'confusion_matrix_rf.png'), dpi=300)
plt.close()

print("\nAnalysis complete! All visual artifacts generated cleanly.")
