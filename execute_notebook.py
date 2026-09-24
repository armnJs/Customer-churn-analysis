import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb_path = r"d:\Armaan\Data visualization\Customer churn analysis\Customer Churn analysis.ipynb"

print(f"Reading notebook from {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

print("Executing notebook cells top to bottom...")
try:
    ep.preprocess(nb, {'metadata': {'path': r"d:\Armaan\Data visualization\Customer churn analysis"}})
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error executing notebook: {e}")
    raise e

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Executed notebook saved with cell outputs to {nb_path}")
