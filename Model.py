from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)


df = pd.read_csv('dataset/*******.csv')

###############################################################
# # Tüm modeller (Decision Tress & Linear Models) df_scale ile
###############################################################

def compare_models(X, y, random_state=42, cv=5):

    # Modeller
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "KNN" : KNeighborsRegressor(),
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


X = df.drop(['NEW_Price_log','Price(TL)'], axis=1)
y = df["NEW_Price_log"]

results = compare_models(X, y)
print(results)

#               Model  R2 Score  MAE (log)  RMSE (log)
# 0           CatBoost    0.8471       0.15        0.24
# 1           LightGBM    0.8419       0.16        0.24
# 2  Linear Regression    0.8213       0.17        0.23
# 3      Random Forest    0.8211       0.17        0.25
# 4   Ridge Regression    0.8206       0.17        0.23
# 5  Gradient Boosting    0.8159       0.17        0.26
# 6                KNN    0.6874       0.24        0.33
# 7   Lasso Regression    0.6305       0.26        0.38

results['real_rmse'] = np.expm1(results['RMSE (log)'])


################################################
# Sadece Ağaç Modelleri (Decision Tress) df ile
################################################


def compare_models(X, y, random_state=42, cv=5):

    # Modeller
    models = {
        #"Linear Regression": LinearRegression(),
        #"Ridge Regression": Ridge(alpha=1.0),
        #"Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        #"Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
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



X = df.drop(['NEW_Price_log','Price(TL)'], axis=1)
y = df["NEW_Price_log"]

results = compare_models(X, y)
print(results)

#            Model  R2 Score  MAE (log)  RMSE (log)
# 0       CatBoost    0.8480       0.15        0.24
# 1       LightGBM    0.8431       0.16        0.24
# 2  Random Forest    0.8219       0.17        0.25



###########################################
# Model Seçimi
###########################################
catboost_model = CatBoostRegressor(verbose=False, random_state=17)

X_train, X_val, y_train, y_val = train_test_split(X,y , test_size=0.20, random_state=17)

##  RandomizedSearchCV Kullanımı
param_dist = {
    'learning_rate': [0.01, 0.05, 0.08, 0.09, 0.1],
    'depth': [4, 6, 8, 10, 11 ,14],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [400, 500, 600, 700, 900, 1000, 1100],
    'bagging_temperature': [0, 0.5, 1],
}

random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train,y_train)
print(random_search.best_params_)
print(random_search.best_score_)
# 0.9552958071105655

# {'learning_rate': 0.05,
# 'l2_leaf_reg': 7,
# 'iterations': 900,
# 'depth': 6,
# 'bagging_temperature': 1}


cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='r2')
print("Cross-validation R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())

# Cross-validation R2 scores: [0.58043954 0.91483785 0.8714151  0.92267331 0.90523352]
# Mean CV R2: 0.8389198630862136


##  GridSearchCV Kullanımı
grid_params = {
    'learning_rate': [0.04, 0.05, 0.06, 0.7],
    'depth': [6, 7, 8, 9],
    'l2_leaf_reg': [6, 7, 8, 9],
    'iterations': [800, 900, 950, 1000]
}

grid_search = GridSearchCV(
    estimator=catboost_model,
    param_grid=grid_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
# {'depth': 6, 'iterations': 950, 'l2_leaf_reg': 7, 'learning_rate': 0.06}




###########################################
# Final Model Kurulumu
###########################################

Final_model = catboost_model.set_params(**grid_search.best_params_).fit(X_train, y_train)

# Model tahmini (log formundaki tahmin)
y_pred_log = Final_model.predict(X_val)

# Gerçek değerler ve tahminleri log formundan çıkar (Price TL olarak)
y_val_real = np.expm1(y_val)
y_pred_real = np.expm1(y_pred_log)

# RMSE: Gerçek TL cinsinden
rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))

# R2: Gerçek TL cinsinden
r2 = r2_score(y_val_real, y_pred_real)

# train-validation hata oranı
print(f"Validation R2 (actual TL):{r2: .4f}")
print(f"Validation RMSE (actual TL): {rmse: .4f}")


# Şimdi train deki hata oranı
y_train_pred_log = Final_model.predict(X_train)
y_train_pred_real = np.expm1(y_train_pred_log)
y_train_real = np.expm1(y_train)

# R2 ve RMSE
r2_train = r2_score(y_train_real, y_train_pred_real)
rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))

print(f"Train R2 (actual TL): {r2_train: .4f}")
print(f"Train RMSE (actual TL):{rmse_train: .4f}")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(Final_model, X_train)


################################
# Model Kaydetme
################################
feature_names = X_train.columns.tolist()
with open("pickle_files/model_features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("pickle_files/final_catboost_model.pkl", "wb") as file:
    pickle.dump(Final_model, file)
