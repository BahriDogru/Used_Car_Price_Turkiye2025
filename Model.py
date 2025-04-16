from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)

df = pd.read_csv('dataset/Model_data.csv')
df.head()
df.drop(['DamageCost', 'VehicleTax()', 'Kilometers', 'Price(TL)', 'ListingDate'], axis=1, inplace=True)


def compare_models(X, y, random_state=42, cv=5):

    # Modeller
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        #"XGBoost" : XGBRegressor(random_state=random_state),
        "LightGBM" : LGBMRegressor(random_state=random_state, verbose=-1),
        "CatBoost" : CatBoostRegressor(verbose=False, random_state=random_state)
    }

    results = []

    for name, model in models.items():

        mae = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv).mean()
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv).mean())
        r2 = cross_val_score(model, X, y, scoring='r2', cv=cv).mean()


        results.append({
            'Model': name,
            'R2 Score': round(r2, 4),
            'MAE (log)': round(mae, 2),
            'RMSE (log)': round(rmse, 2)
        })

    return pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)


# Örnek kullanım
X = df.drop("Price_log", axis=1)
y = df["Price_log"]

results = compare_models(X, y)
print(results)

#          Model  R2 Score  MAE (log)  RMSE (log)
# 0       LightGBM    0.7615       0.20        0.30
# 1       CatBoost    0.7391       0.21        0.32
# 2  Random Forest    0.7308       0.22        0.32

results['real_mae'] = np.expm1(results['MAE (log)'])


###########################################################################
# Eksik değerlerin doldurrularak diğer modellerin de kullanılması
###########################################################################


X = df.drop("Price_log", axis=1)
y = df["Price_log"]

results2 = compare_models(X, y)
print(results2)

results2['real_mae'] = np.expm1(results2['MAE (log)'])

#               Model  R2 Score  MAE (log)  RMSE (log)
# 0           LightGBM    0.7614       0.20        0.30
# 1   Lasso Regression    0.7559       0.21        0.27
# 2           CatBoost    0.7432       0.21        0.32
# 3      Random Forest    0.7312       0.22        0.32
# 4  Gradient Boosting    0.7096       0.22        0.34
# 5   Ridge Regression    0.5468       0.30        0.39
# 6  Linear Regression    0.5298       0.30        0.40






