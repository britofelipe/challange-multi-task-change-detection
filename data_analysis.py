import geopandas as gpd

# Load dataset
file_path = "train.geojson" 
df = gpd.read_file(file_path)

# Print dataset overview
print("\n" + "=" * 60)
print(" 🗺️  DATASET PREVIEW (First 5 Rows)")
print("=" * 60)
print(df.head(), "\n")

# Check for missing values
missing_values = df.isnull().sum()
print("=" * 60)
print(" ❌ MISSING VALUES REPORT")
print("=" * 60)
if missing_values.sum() == 0:
    print("✅ No missing values found in the dataset.\n")
else:
    print(missing_values[missing_values > 0].to_string(), "\n")

# Check data types of the columns
print("=" * 60)
print(" 🔍 COLUMN DATA TYPES")
print("=" * 60)
print(df.dtypes, "\n")

# Explore categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("=" * 60)
print(" 📊 CATEGORICAL COLUMN VALUES")
print("=" * 60)
for col in categorical_cols:
    print(f"🟢 Column: **{col}**")
    print("Unique Values:", df[col].unique())
    print("-" * 60)

# Extract area and perimeter from geometry
df["area"] = df["geometry"].area
df["perimeter"] = df["geometry"].length

# Drop original geometry column
