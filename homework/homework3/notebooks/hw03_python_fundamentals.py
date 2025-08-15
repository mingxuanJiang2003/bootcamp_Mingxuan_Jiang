import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

# 1. NumPy Operations
# Create a random integer array
arr = np.random.randint(1, 100, size=10)
print("Original array:", arr)

# Element-wise operations
arr_plus_10 = arr + 10
arr_squared = arr ** 2
print("Array + 10:", arr_plus_10)
print("Array squared:", arr_squared)

# Compare loop execution vs vectorized execution
loop_start = time.time()
arr_squared_loop = [x**2 for x in arr]
loop_end = time.time()

vec_start = time.time()
arr_squared_vec = arr ** 2
vec_end = time.time()

print(f"Loop time: {loop_end - loop_start:.8f} seconds")
print(f"Vectorized time: {vec_end - vec_start:.8f} seconds")

# 2. Dataset Loading
# Create a mock dataset and save as CSV
os.makedirs("data", exist_ok=True)
df_mock = pd.DataFrame({
    "Category": np.random.choice(["A", "B", "C"], size=20),
    "Value1": np.random.randint(10, 100, size=20),
    "Value2": np.random.randint(100, 200, size=20)
})
df_mock.to_csv("data/starter_data.csv", index=False)

# Load CSV using pandas
df = pd.read_csv("data/starter_data.csv")
print("\nData Info:")
print(df.info())
print("\nData Head:")
print(df.head())

# 3. Summary Statistics
# Basic descriptive statistics
summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Group by Category and calculate mean
grouped_stats = df.groupby("Category").agg({
    "Value1": "mean",
    "Value2": "mean"
})
print("\nGrouped by Category (Mean):")
print(grouped_stats)

# 4. Save Outputs
# Save summary statistics to CSV
os.makedirs("data/processed", exist_ok=True)
summary_stats.to_csv("data/processed/summary.csv")

# Create and save a simple bar plot
plt.figure(figsize=(6,4))
grouped_stats.plot

