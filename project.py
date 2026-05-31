import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

# Show first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print("\nAfter Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

# Visualization 1
plt.figure(figsize=(8,5))
sns.histplot(df[numeric_cols[0]], kde=True)
plt.title("Distribution Plot")
plt.show()

# Visualization 2
if len(numeric_cols) >= 2:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
    plt.title("Scatter Plot")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nProject Completed Successfully!")