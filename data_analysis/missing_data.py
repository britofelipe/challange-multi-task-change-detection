import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "train.geojson"
df = gpd.read_file(file_path)

# Calculate missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

# Filter only columns with missing data
missing_report = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_report = missing_report[missing_report["Missing Values"] > 0].sort_values(by="Percentage", ascending=False)

# Print summary
print("=" * 60)
print(" ❌ MISSING VALUES REPORT")
print("=" * 60)
if missing_report.empty:
    print("✅ No missing values found in the dataset.\n")
else:
    print(missing_report, "\n")

# Visualizing missing data
plt.figure(figsize=(10, 6))
sns.barplot(y=missing_report.index, x=missing_report['Percentage'], palette='Reds_r')
plt.xlabel("Percentage of Missing Values")
plt.title("Missing Data Distribution")
plt.show()
