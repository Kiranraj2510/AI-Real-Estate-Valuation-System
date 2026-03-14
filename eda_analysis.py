import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("housing.csv")

# Show dataset information
print(df.head())
print(df.describe())

# Fill missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# 1 Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['median_house_value'], bins=50)
plt.title("House Price Distribution")
plt.show()

# 2 Income vs House price
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['median_income'], y=df['median_house_value'])
plt.title("Median Income vs House Price")
plt.show()

# 3 Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()