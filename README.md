# codealpha_sales-pridiction-using-python
Sales Prediction using Python ● Predict future sales based on factors like advertising spend, target segment and platform. ● Prepare data through cleaning, transformation and feature selection. ● Use regression or time series models to forecast sales. ● Analyze how changes in advertising impact sales outcomes. 
# Sales Prediction using Python
# -----------------------------
# Features:
# - Load or generate data
# - Data cleaning & transformation
# - Feature selection
# - Regression model to predict sales
# - Impact analysis of advertising spend
# - Visualization & insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# 1. Create or Load Data
# ----------------------

def generate_sales_data(n=200, seed=42):
    np.random.seed(seed)
    advertising_spend = np.random.randint(1000, 10000, size=n)  # in $
    target_segment = np.random.choice(['Teens', 'Adults', 'Seniors'], size=n)
    platform = np.random.choice(['Online', 'TV', 'Print'], size=n)

    # Base sales with influence from advertising and platform
    base_sales = 5000 + 0.8 * advertising_spend
    platform_effect = np.where(platform == 'Online', 2000,
                        np.where(platform == 'TV', 1500, 800))
    segment_effect = np.where(target_segment == 'Teens', 1000,
                        np.where(target_segment == 'Adults', 1500, 500))

    noise = np.random.normal(0, 2000, n)  # random variation

    sales = base_sales + platform_effect + segment_effect + noise

    df = pd.DataFrame({
        'AdvertisingSpend': advertising_spend,
        'TargetSegment': target_segment,
        'Platform': platform,
        'Sales': np.round(sales, 2)
    })
    return df

# Change to your CSV path if you have real data
data_path = None  # Example: "sales_data.csv"

if data_path:
    df = pd.read_csv(data_path)
else:
    df = generate_sales_data()

print("First 5 rows of data:")
print(df.head())

# ----------------------
# 2. Data Cleaning & Transformation
# ----------------------

# Handle missing values
df.dropna(inplace=True)

# Convert categorical variables to numeric (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['TargetSegment', 'Platform'], drop_first=True)

# ----------------------
# 3. Feature Selection
# ----------------------
X = df_encoded.drop('Sales', axis=1)
y = df_encoded['Sales']

# ----------------------
# 4. Train-Test Split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# 5. Model Training
# ----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------
# 6. Predictions & Evaluation
# ----------------------
y_pred = model.predict(X_test)

print("\nModel Performance:")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ----------------------
# 7. Impact Analysis of Advertising Spend
# ----------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Impact on Sales:")
print(coef_df)

# ----------------------
# 8. Visualization
# ----------------------

# Actual vs Predicted Sales
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# Advertising Spend vs Sales
plt.figure(figsize=(6, 5))
plt.scatter(df['AdvertisingSpend'], df['Sales'], alpha=0.6)
plt.xlabel("Advertising Spend ($)")
plt.ylabel("Sales ($)")
plt.title("Advertising Spend vs Sales")
plt.grid(True)
plt.show()

# ----------------------
# 9. Business Insights
# ----------------------
print("\nBusiness Insights:")
print("- Advertising spend shows a strong positive relationship with sales.")
print("- Online and TV platforms tend to drive higher sales compared to Print.")
print("- Targeting Adults and Teens yields better results than Seniors.")
print("- Increasing ad spend strategically on high-performing platforms could maximize ROI.")

# Save processed dataset
df_encoded.to_csv("processed_sales_data.csv", index=False)
print("\nProcessed dataset saved as 'processed_sales_data.csv'")

