
# the model explains ~98% of the variance

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


df = pd.read_csv("zameen.csv")

# Step 2: Drop irrelevant columns
df.drop(columns=['agency', 'agent', 'page_url', 'property_id'], inplace=True, errors='ignore')

# Step 3: Clean 'date_added'
df = df[~df['date_added'].astype(str).str.contains('#')]  # remove '########'
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df.dropna(subset=['date_added'], inplace=True)

# Step 4: Extract date features
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['day_added'] = df['date_added'].dt.day
df.drop(columns=['date_added'], inplace=True)

# Step 5: Convert Area Type + Size into total_area_marla
def convert_to_marla(size, unit):
    if pd.isna(size) or pd.isna(unit):
        return None
    unit = str(unit).strip().lower()
    try:
        size = float(size)
    except:
        return None

    if unit == "marla":
        return size
    elif unit == "kanal":
        return size * 20
    elif unit in ["sq. yards", "square yards"]:
        return size / 30.25
    elif unit in ["sq. ft.", "square feet"]:
        return size / 225
    else:
        return None

# use log for target when prices , salary are highly skewed to normalize the distribution
df['total_area_marla'] = df.apply(lambda row: convert_to_marla(row['Area Size'], row['Area Type']), axis=1)
df.drop(columns=['Area Size', 'Area Type', 'area', 'Area Category'], inplace=True, errors='ignore')
df.dropna(subset=['total_area_marla'], inplace=True)
df.drop(columns=['location_id'], inplace=True, errors='ignore')

# Step 6: Encode categorical variables
cat_cols = ['property_type', 'location', 'city', 'province_name', 'purpose']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Step 7: Feature & target split
X = df.drop(columns=['price'])
y = np.log1p(df['price'])  # log (price + 1)

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train model
#n_estimators=350,max_depth=19,min_samples_leaf=3,
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 10: Predict & evaluate
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {mse**0.5:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 11: Feature importance plot
importances = model.feature_importances_
feature_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.show()

import joblib



feature_order = X.columns.tolist()  # <- exact column order used in training
joblib.dump((model, feature_order), 'price_model.pkl')
print("✅ Model and feature order saved successfully!")


#model = joblib.load('price_model.pkl')
#print("✅ Model loaded successfully")