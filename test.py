import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import math
pd.set_option('display.float_format', '{:.0f}'.format)
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)

# Kaydedilen model ve özellikleri yükle
model = joblib.load("pickle_files/final_catboost_model.pkl")
train_columns = joblib.load("pickle_files/model_features.pkl")


# Test verisini oku
test_df = pd.read_csv("dataset/test_data.csv")

# Target değişkenlerini yedekle
true_price_log = test_df["NEW_Price(TL)_log"].copy()
true_price_tl = test_df['Price(TL)'].copy()

# Modelin eğitimde kullandığı feature'ları test verisinde olmayanları sıfırla doldur
for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0

# Fazla kolonları çıkar
test_df = test_df[train_columns]


# Log dönüşümlü fiyat tahminlerini yap
predicted_price_log = model.predict(test_df)

# Gerçek para birimine çevir (log -> TL)
predicted_price_tl = np.expm1(predicted_price_log)


results_df = pd.DataFrame({
    "True_price_Tl" : true_price_tl.round(2),
    "Predicted_Price_TL": predicted_price_tl.round(2),
    "Actual_Log": true_price_log,
    "Predicted_Log": predicted_price_log.round(4)
})


# İlk 10 satırı görelim
print(results_df.head(30))

###########################################
# Test hata oranlarını hesapla
###########################################

predicted_prices = np.expm1(predicted_price_log)
true_prices = np.expm1(true_price_log)

# R2 ve RMSE hesapla
test_r2 = r2_score(true_prices, predicted_prices)
test_rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))

actual_r2 = r2_score(true_price_tl , predicted_price_tl)
actual_rmse = np.sqrt(mean_squared_error(true_price_tl, predicted_price_tl))


print(f"Test R2 (actual TL): {actual_r2:.4f}")
print(f"Test RMSE (actual TL): {actual_rmse:,.2f} TL")
print(f"Test R2 (actual TL): {test_r2:.4f}")
print(f"Test RMSE (actual TL): {test_rmse:,.2f} TL")

