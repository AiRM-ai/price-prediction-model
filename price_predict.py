from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

df = pd.read_csv('retail_price.csv', delimiter=',')
df['month_year'] = pd.to_datetime(df['month_year'], format='%m/%d/%Y')
df = df.sort_values(by=['product_id', 'month_year']).reset_index(drop=True)

TARGET = 'qty'
CATEGORICAL_FEATURES = ['product_id', 'product_category_name']
df['month'] = df['month_year'].dt.month
df['year'] = df['month_year'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df = df.drop(columns=['month_year', 'month'])
df = df.fillna(0)

X = df.drop(columns=[TARGET, 'total_price'])
y = df[TARGET]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = X_train.copy()
X_train_encoded[CATEGORICAL_FEATURES] = encoder.fit_transform(X_train_encoded[CATEGORICAL_FEATURES])


model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train_encoded, y_train)


df_orig = pd.read_csv('retail_price.csv', delimiter=',')
df_orig['month_year'] = pd.to_datetime(df_orig['month_year'], format='%m/%d/%Y')
latest_date_per_product = df_orig.loc[df_orig.groupby('product_id')['month_year'].idxmax()]
base_df = df.loc[df.index.isin(latest_date_per_product.index)].reset_index(drop=True)

base_df = base_df.drop_duplicates(subset=['product_id'], keep='first')

price_multipliers = np.linspace(0.8, 1.2, 21)
all_optimization_data = []
for index, row in base_df.iterrows():
    original_price = row['unit_price']
    for multiplier in price_multipliers:
        new_row = row.copy()
        new_row['unit_price'] = original_price * multiplier
        new_row['original_price'] = original_price
        all_optimization_data.append(new_row)

optimization_df = pd.DataFrame(all_optimization_data)


X_columns = X.columns.tolist()
X_opt_base = optimization_df[X_columns].copy()
X_opt_encoded = X_opt_base.copy()
X_opt_encoded[CATEGORICAL_FEATURES] = encoder.transform(X_opt_encoded[CATEGORICAL_FEATURES])

optimization_df['predicted_qty'] = model.predict(X_opt_encoded)
optimization_df['predicted_revenue'] = optimization_df['unit_price'] * optimization_df['predicted_qty']


optimized_prices_df = optimization_df.loc[optimization_df.groupby('product_id')['predicted_revenue'].idxmax()]


result_json = optimized_prices_df[['product_id', 'product_category_name', 'unit_price', 'predicted_qty', 'predicted_revenue', 'original_price']].rename(columns={'unit_price': 'optimal_price'})
result_json['optimal_price'] = result_json['optimal_price'].round(2)
result_json['predicted_qty'] = result_json['predicted_qty'].round(2)
result_json['predicted_revenue'] = result_json['predicted_revenue'].round(2)
result_json = result_json.drop_duplicates(subset=['product_id'], keep='first')


result_json.to_json('optimized_price_predictions_unique_v2.json', orient='records', indent=4)

# y_pred = model.predict(X_test_encoded)

# # 2. Calculate Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)