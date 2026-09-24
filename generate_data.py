import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 2500

customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
age = np.random.randint(18, 68, size=n_samples)
gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.49, 0.51])
tenure = np.random.randint(1, 60, size=n_samples)
usage_freq = np.random.randint(1, 30, size=n_samples)
support_calls = np.random.poisson(lam=2.5, size=n_samples)
payment_delay = np.random.exponential(scale=7, size=n_samples).astype(int)
subscription_type = np.random.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.40, 0.35, 0.25])
contract_length = np.random.choice(['Monthly', 'Quarterly', 'Annual'], size=n_samples, p=[0.50, 0.30, 0.20])
total_spend = np.round(tenure * np.random.uniform(50, 150, size=n_samples) + np.random.uniform(50, 500, size=n_samples), 2)
last_interaction = np.random.randint(1, 30, size=n_samples)

# Calculate Realistic Churn Probability Logit
logit = (
    -1.5 
    + 0.12 * payment_delay 
    + 0.45 * (support_calls - 2) 
    - 0.04 * tenure 
    + 0.8 * (contract_length == 'Monthly') 
    - 0.5 * (contract_length == 'Annual') 
    - 0.3 * (subscription_type == 'Premium')
)

prob = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(n_samples) < prob).astype(int)

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': age,
    'Gender': gender,
    'Tenure': tenure,
    'Usage Frequency': usage_freq,
    'Support Calls': support_calls,
    'Payment Delay': payment_delay,
    'Subscription Type': subscription_type,
    'Contract Length': contract_length,
    'Total Spend': total_spend,
    'Last Interaction': last_interaction,
    'Churn': churn
})

# Insert missing row at the end to demonstrate data cleaning in notebook
df.iloc[n_samples - 1] = [np.nan] * 12

file_path = r"d:\Armaan\Data visualization\Customer churn analysis\customer.csv"
df.to_csv(file_path, index=False)
print(f"Successfully generated realistic customer churn dataset ({n_samples} rows) at {file_path}")
print(f"Overall Churn Rate: {df['Churn'].dropna().mean()*100:.2f}%")
