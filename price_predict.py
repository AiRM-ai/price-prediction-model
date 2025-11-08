import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
# import kagglehub

# comp_1, comp_2, comp_3: The prices of Competitor 1, Competitor 2, and Competitor 3 for the same product in a given period (e.g., month).
# ps1, ps2, ps3: The product scores or ratings associated with the products from Competitor 1, Competitor 2, and Competitor 3, respectively.
# fp1, fp2, fp3: The freight (shipping) prices or costs associated with the products from Competitor 1, Competitor 2, and Competitor 3, respectively.
# lag_price: The lagged price of the primary product, which is the product's price from a previous time period (e.g., the prior month's price). This is used for time-series analysis or forecasting
df = pd.read_csv('C:/Users/alisa/Documents/AiRM-prediction-model/price-prediction-model/retail_price.csv', delimiter=',')

# Convert the month_year column to datetime objects
df['month_year'] = pd.to_datetime(df['month_year'], format='%m/%d/%Y') # Adjust format if needed

# IMPORTANT: Sort the data by product and then by time. 
# This is mandatory for creating accurate time-based (lag) features.
df = df.sort_values(by=['product_id', 'month_year']).reset_index(drop=True)


TARGET = 'qty'
CATEGORICAL_FEATURES = ['product_id', 'product_category_name']

# --- 1. Create Lagged Features (Demand Inertia) ---

# Lagged demand (qty from the previous time step for the same product)
df['lag_qty'] = df.groupby('product_id')[TARGET].shift(1)

# Lagged price change (captures immediate reaction to a price change)
df['price_change'] = df.groupby('product_id')['unit_price'].diff().fillna(0) # 'diff' calculates the difference

df['price_vs_comp1'] = df['unit_price'] / df['comp_1']

df['price_vs_ps1'] = df['unit_price'] / df['ps1']

df['freight_ratio'] = df['freight_price'] / (df['unit_price'] + 1e-6)


df['avg_comp_price'] = df[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)



df['month'] = df['month_year'].dt.month
df['year'] = df['month_year'].dt.year

# Cyclical Encoding for Month (better for models than raw 1-12)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df = df.drop(columns=['month_year', 'month'])


df = df.fillna(0) 


X = df.drop(columns=[TARGET, 'total_price']) 
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



encoder = TargetEncoder(cols=CATEGORICAL_FEATURES)


X_train_encoded = encoder.fit_transform(X_train, y_train) 


X_test_encoded = encoder.transform(X_test)



import xgboost as xgb

model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)

mean_qty = y_test.mean()
std_qty = y_test.std()
rmse = np.sqrt(115.36553955078125)

print(f"Average Demand (y_test mean): {mean_qty:.2f}")
print(f"Standard Deviation of Demand (y_test std): {std_qty:.2f}")
print(f"Model RMSE: {rmse:.2f}")

#THIS MODEL IS SO BAD I NEED TO WORK DW ABOUT IT LOOOOOL
# Average Demand (y_test mean): 15.55
# Standard Deviation of Demand (y_test std): 17.68
# Model RMSE: 10.74
