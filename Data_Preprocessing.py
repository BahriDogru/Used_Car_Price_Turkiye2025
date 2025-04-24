import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)
pd.set_option('display.float_format', '{:.0f}'.format) # çıktılarda sayıları ondalıklı kısmı olmadan göstermeni sağlar.
# pd.reset_option('display.float_format') # ondalıklı gösterim için bu kullanılır.

df = pd.read_csv("dataset/train_data.csv", encoding="ISO-8859-9")
df.columns = ['Title', 'Address', 'City', 'Price(TL)', 'ListingID', 'ListingDate', 'Brand', 'Series', 'Model', 'Year', 'Kilometers', 'GearType', 'FuelType', 'BodyType', 'Color', 'EngineSize', 'EnginePower', 'DriveType', 'PaintAndPartsCondition', 'TradeInStatus', 'SellerType', 'VehicleTax(TL)', 'TramerCondition']
df.info()
df.head()

# Tüm sütunlardaki string değerlerin başındaki ve sonundaki boşlukları \n, \r, \t gibi karakterler temizle
def clean_text_columns(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).apply(lambda x: x.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip())
    return df

df = clean_text_columns(df)

# remove Title and ListteningID
drop_list = ['Title', 'ListingID']
df = df.drop(drop_list, axis=1)
df = df.drop_duplicates()
######################
## Dtype düzeltme
######################

# 'Price(TL)' sütununu temizle ve float'a çevir
df['Price(TL)'] = df['Price(TL)'].str.replace('.', '', regex=False)  # binlik ayıracı olan noktaları kaldır
df['Price(TL)'] = df['Price(TL)'].str.replace(' TL', '', regex=False)  # ' TL' kısmını kaldır
df['Price(TL)'] = df['Price(TL)'].str.replace(',', '.', regex=False)  # eğer ondalık virgül varsa noktaya çevir
df['Price(TL)'] = pd.to_numeric(df['Price(TL)'], errors='coerce')

# Year değişkenini sayısala döndürme
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# ListininDate
month_map = {
    "Ocak": "January", "Şubat": "February", "Mart": "March", "Nisan": "April",
    "Mayıs": "May", "Haziran": "June", "Temmuz": "July", "Ağustos": "August",
    "Eylül": "September", "Ekim": "October", "Kasım": "November", "Aralık": "December"
} # Türkçe → İngilizce ay çevirme
# Ay adlarını İngilizce’ye çevirme
for tr, en in month_map.items():
    df['ListingDate'] = df['ListingDate'].str.replace(tr, en)

# ListininDate sütununu datetime formatına çevirme
df['ListingDate'] = df['ListingDate'].str.strip()
df['ListingDate'] = df['ListingDate'].str.replace(r"\s+", " ", regex=True)
df['ListingDate'] = pd.to_datetime(df['ListingDate'], format='%d %B %Y')


# Kilometers sütununu temizleme ve sayısal'a çevirme
df["Kilometers"] = df["Kilometers"].str.strip(" km")
df['Kilometers'] = df['Kilometers'].str.replace('.', '', regex=False)
df['Kilometers'] = pd.to_numeric(df['Kilometers'], errors='coerce')


# EngineSize ve EnginePower sütunlarını temizle ne sayısala çevirme
def extract_number_fixed(value):
    # Boş veya NaN değerleri string'e çevir
    value = str(value).lower()

    # Temizlenmesi gereken karakterleri kaldır
    value = value.replace("'", "").replace("cm3", "").replace("cc", "").replace("hp", "")

    # Sayı aralıklarını yakala
    nums = re.findall(r'\d+', value)

    if len(nums) == 1:
        return float(nums[0])
    elif len(nums) == 2:
        return round((int(nums[0]) + int(nums[1])) / 2)
    else:
        return np.nan
df['EngineSize'] = df['EngineSize'].apply(extract_number_fixed)
df['EnginePower'] = df['EnginePower'].apply(extract_number_fixed)

df.info()
###########################
# Edit Variables - Missing Values
##########################
df.isna().sum()
def remove_nan_col(dataframe):
    """
      Veri setindeki her gözlemdeki (satırdaki) NaN değerlerini sayar ve
      bir gözlemdeki NaN değerlerinin sayısı sütun sayısının yarısından fazla veya eşitse
      o gözlemi veri setinden çıkarır.

      Args:
        df: Pandas DataFrame.

      Returns:
        new_df: NaN değerleri belirlenen değerden daha az olan verilerin bulunduğu dataframe.
      """
    count_col = dataframe.shape[1]
    thr_value = count_col / 2
    drop_rows = []

    for index, row in dataframe.iterrows():
        nan_count = row.isna().sum()
        if nan_count >= thr_value:
            drop_rows.append(index)

    new_df = dataframe.drop(drop_rows)
    return new_df

df = remove_nan_col(df)


# BodyType
list_BodyType = ['Hatchback/5', 'Sedan', 'Coupe', 'Hatchback/3','Roadster','Station wagon','MPV']
df["BodyType"] = df["BodyType"].where(df["BodyType"].isin(list_BodyType), np.nan)
df['BodyType'] = df.groupby(['Brand', 'Series', 'Model'])['BodyType']\
                   .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
df['BodyType'].fillna(df['BodyType'].mode()[0], inplace=True)

# DriveType
list_DriveType = ['Önden Çekiş', '4WD (Sürekli)','Arkadan İtiş']
df["DriveType"] = df["DriveType"].where(df["DriveType"].isin(list_DriveType), np.nan)
df['DriveType'] = df.groupby(['Brand', 'Series', 'Model'])['DriveType']\
                   .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
df['DriveType'].fillna(df['DriveType'].mode()[0], inplace=True)


# FuelType
list_FuelType = ['Benzin', 'Dizel', 'Hibrit', 'LPG & Benzin','Elektrik']
df["FuelType"] = df["FuelType"].where(df["FuelType"].isin(list_FuelType), np.nan)
df['FuelType'] = df.groupby(['Brand', 'Series', 'Model'])['FuelType']\
                   .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
df['FuelType'].fillna(df['FuelType'].mode()[0], inplace=True)

# SellerType
list_SellerType = ['Belirtilmemiş', 'Galeriden', 'Sahibinden', 'Yetkili Bayiden']
df["SellerType"] = df["SellerType"].where(df["SellerType"].isin(list_SellerType), np.nan)
df['SellerType'].fillna('Belirtilmemiş', inplace=True)

# TradeInStatus
list_TradeInStatus = ['Takasa Uygun','Takasa Uygun Değil','Belirtilmemiş']
df["TradeInStatus"] = df["TradeInStatus"].where(df["TradeInStatus"].isin(list_TradeInStatus), np.nan)
df['TradeInStatus'].fillna('Belirtilmemiş', inplace=True)

# EngineSize
df["EngineSize"] = df.groupby(["Brand", "Model"])["EngineSize"].transform(lambda x: x.fillna(x.median()))
df['EngineSize'].fillna(df['EngineSize'].median(), inplace=True)

# EnginePower
df["EnginePower"] = df.groupby(["Brand", "Model"])["EnginePower"].transform(lambda x: x.fillna(x.median()))
df['EnginePower'].fillna(df['EnginePower'].median(), inplace=True)


# VehicleTax() ---> Missing Values
def calculate_mtv(engine_size, vehicle_year, price):
    """
    Aracın motor hacmi, yaşı ve değerine göre MTV'sini hesaplar.
    Kaynak olarak belirtilen web sitesindeki 2025 tarifesi baz alınmıştır.
    MTV hesaplaması yapılırken, araç tipinin "otomobil" olduğu varsayılmıştır.
    Kaynak Link: https://www.hedeffilo.com/blog/motorlu-tasitlar-vergisi-ve-mtv-oranlari-mtv-rehberi#:~:text=MTV%20hesaplamas%C4%B1%20arac%C4%B1n%20cinsi%2C%20ya%C5%9F%C4%B1,y%C4%B1lda%20iki%20defa%20%C3%B6denmesi%20gerekiyor.
    """
    # Aracın yaşını hesaplama
    current_year = datetime.now().year
    try:
        vehicle_age = current_year - int(vehicle_year)
        engine_size = float(engine_size)  # EngineSize_clean'i float'a dönüştürmeyi dene
        price = float(price)  # Price'ı float'a dönüştürmeyi dene
    except ValueError:
        return np.nan
    mtv = 0  # Başlangıç değeri atandı

    # 2018 öncesi araçlar için MTV hesaplaması
    if vehicle_year < 2018:
        if engine_size <= 1300:
            if vehicle_age <= 3:
                mtv = 4834
            elif 4 <= vehicle_age <= 6:
                mtv = 3372
            elif 7 <= vehicle_age <= 11:
                mtv = 71882
            elif 12 <= vehicle_age <= 15:
                mtv = 1420
            elif vehicle_age > 15:
                mtv = 499
        elif 1301 <= engine_size <= 1600:
            if vehicle_age <= 3:
                mtv = 8421
            elif 4 <= vehicle_age <= 6:
                mtv = 6314
            elif 7 <= vehicle_age <= 11:
                mtv = 3661
            elif 12 <= vehicle_age <= 15:
                mtv = 2587
            elif vehicle_age > 15:
                mtv = 993
        elif 1601 <= engine_size <= 1800:
            if vehicle_age <= 3:
                mtv = 14885
            elif 4 <= vehicle_age <= 6:
                mtv = 11626
            elif 7 <= vehicle_age <= 11:
                mtv = 6848
            elif 12 <= vehicle_age <= 15:
                mtv = 4168
            elif vehicle_age > 15:
                mtv = 1612
        elif 1801 <= engine_size <= 2000:
            if vehicle_age <= 3:
                mtv = 23454
            elif 4 <= vehicle_age <= 6:
                mtv = 18057
            elif 7 <= vehicle_age <= 11:
                mtv = 10613
            elif 12 <= vehicle_age <= 15:
                mtv = 6314
            elif vehicle_age > 15:
                mtv = 2487
        elif 2001 <= engine_size <= 2500:
            if vehicle_age <= 3:
                mtv = 35175
            elif 4 <= vehicle_age <= 6:
                mtv = 25534
            elif 7 <= vehicle_age <= 11:
                mtv = 15954
            elif 12 <= vehicle_age <= 15:
                mtv = 9528
            elif vehicle_age > 15:
                mtv = 3766
        elif 2501 <= engine_size <= 3000:
            if vehicle_age <= 3:
                mtv = 49052
            elif 4 <= vehicle_age <= 6:
                mtv = 42669
            elif 7 <= vehicle_age <= 11:
                mtv = 26654
            elif 12 <= vehicle_age <= 15:
                mtv = 14329
            elif vehicle_age > 15:
                mtv = 5259
        elif 3001 <= engine_size <= 3500:
            if vehicle_age <= 3:
                mtv = 74703
            elif 4 <= vehicle_age <= 6:
                mtv = 66218
            elif 7 <= vehicle_age <= 11:
                mtv = 40486
            elif 12 <= vehicle_age <= 15:
                mtv = 20203
            elif vehicle_age > 15:
                mtv = 7409
        elif 3501 <= engine_size <= 4000:
            if vehicle_age <= 3:
                mtv = 117462
            elif 4 <= vehicle_age <= 6:
                mtv = 101427
            elif 7 <= vehicle_age <= 11:
                mtv = 59730
            elif 12 <= vehicle_age <= 15:
                mtv = 26654
            elif vehicle_age > 15:
                mtv = 10613
        elif 4001 <= engine_size:
            if vehicle_age <= 3:
                mtv = 192250
            elif 4 <= vehicle_age <= 6:
                mtv = 144166
            elif 7 <= vehicle_age <= 11:
                mtv = 85377
            elif 12 <= vehicle_age <= 15:
                mtv = 38363
            elif vehicle_age > 15:
                mtv = 14885
    # 2018 sonrası araçlar için MTV hesaplaması (farklı bir tabloya göre)
    else:
        if engine_size <= 1300:
            if price <= 259900:
                if vehicle_age <= 3:
                    mtv = 4834
                elif 4 <= vehicle_age <= 6:
                    mtv = 3372
                elif 7 <= vehicle_age <= 11:
                    mtv = 1882
                elif 12 <= vehicle_age <= 15:
                    mtv = 1420
                else:
                    mtv = 499
            elif 259900 < price <= 455300:
                if vehicle_age <= 3:
                    mtv = 5313
                elif 4 <= vehicle_age <= 6:
                    mtv = 3707
                elif 7 <= vehicle_age <= 11:
                    mtv = 2068
                elif 12 <= vehicle_age <= 15:
                    mtv = 1565
                else:
                    mtv = 551
            else:
                if vehicle_age <= 3:
                    mtv = 5803
                elif 4 <= vehicle_age <= 6:
                    mtv = 4042
                elif 7 <= vehicle_age <= 11:
                    mtv = 2264
                elif 12 <= vehicle_age <= 15:
                    mtv = 1709
                else:
                    mtv = 594
        elif 1301 <= engine_size <= 1600:
            if price <= 259900:
                if vehicle_age <= 3:
                    mtv = 8421
                elif 4 <= vehicle_age <= 6:
                    mtv = 6314
                elif 7 <= vehicle_age <= 11:
                    mtv = 3661
                elif 12 <= vehicle_age <= 15:
                    mtv = 2587
                else:
                    mtv = 993
            elif 259900 < price <= 455300:
                if vehicle_age <= 3:
                    mtv = 9267
                elif 4 <= vehicle_age <= 6:
                    mtv = 6948
                elif 7 <= vehicle_age <= 11:
                    mtv = 4031
                elif 12 <= vehicle_age <= 15:
                    mtv = 2838
                else:
                    mtv = 1085
            else:
                if vehicle_age <= 3:
                    mtv = 10112
                elif 4 <= vehicle_age <= 6:
                    mtv = 7577
                elif 7 <= vehicle_age <= 11:
                    mtv = 4389
                elif 12 <= vehicle_age <= 15:
                    mtv = 3098
                else:
                    mtv = 1184
        elif 1601 <= engine_size <= 1800:
            if price <= 651700:
                if vehicle_age <= 3:
                    mtv = 16370
                elif 4 <= vehicle_age <= 6:
                    mtv = 12801
                elif 7 <= vehicle_age <= 11:
                    mtv = 7523
                elif 12 <= vehicle_age <= 15:
                    mtv = 4589
                else:
                    mtv = 1777
            else:
                if vehicle_age <= 3:
                    mtv = 17876
                elif 4 <= vehicle_age <= 6:
                    mtv = 13956
                elif 7 <= vehicle_age <= 11:
                    mtv = 8218
                elif 12 <= vehicle_age <= 15:
                    mtv = 5014
                else:
                    mtv = 1940
        elif 1801 <= engine_size <= 2000:
            if price <= 651700:
                if vehicle_age <= 3:
                    mtv = 25792
                elif 4 <= vehicle_age <= 6:
                    mtv = 19862
                elif 7 <= vehicle_age <= 11:
                    mtv = 11674
                elif 12 <= vehicle_age <= 15:
                    mtv = 6948
                else:
                    mtv = 2731
            else:
                if vehicle_age <= 3:
                    mtv = 28142
                elif 4 <= vehicle_age <= 6:
                    mtv = 21677
                elif 7 <= vehicle_age <= 11:
                    mtv = 12734
                elif 12 <= vehicle_age <= 15:
                    mtv = 7577
                else:
                    mtv = 2982
        elif 2001 <= engine_size <= 2500:
            if price <= 813900:
                if vehicle_age <= 3:
                    mtv = 38695
                elif 4 <= vehicle_age <= 6:
                    mtv = 28090
                elif 7 <= vehicle_age <= 11:
                    mtv = 17549
                elif 12 <= vehicle_age <= 15:
                    mtv = 10480
                else:
                    mtv = 4145
            else:
                if vehicle_age <= 3:
                    mtv = 42217
                elif 4 <= vehicle_age <= 6:
                    mtv = 30642
                elif 7 <= vehicle_age <= 11:
                    mtv = 19141
                elif 12 <= vehicle_age <= 15:
                    mtv = 11439
                else:
                    mtv = 4522
        elif 2501 <= engine_size <= 3000:
            if price <= 1628900:
                if vehicle_age <= 3:
                    mtv = 53952
                elif 4 <= vehicle_age <= 6:
                    mtv = 48942
                elif 7 <= vehicle_age <= 11:
                    mtv = 29322
                elif 12 <= vehicle_age <= 15:
                    mtv = 15770
                else:
                    mtv = 5780
            else:
                if vehicle_age <= 3:
                    mtv = 58884
                elif 4 <= vehicle_age <= 6:
                    mtv = 51203
                elif 7 <= vehicle_age <= 11:
                    mtv = 31991
                elif 12 <= vehicle_age <= 15:
                    mtv = 17206
                else:
                    mtv = 6308
        elif 3001 <= engine_size <= 3500:
            if price <= 1628900:
                if vehicle_age <= 3:
                    mtv = 82173
                elif 4 <= vehicle_age <= 6:
                    mtv = 73942
                elif 7 <= vehicle_age <= 11:
                    mtv = 44537
                elif 12 <= vehicle_age <= 15:
                    mtv = 22231
                else:
                    mtv = 8142
            else:
                if vehicle_age <= 3:
                    mtv = 89852
                elif 4 <= vehicle_age <= 6:
                    mtv = 80656
                elif 7 <= vehicle_age <= 11:
                    mtv = 48585
                elif 12 <= vehicle_age <= 15:
                    mtv = 24245
                else:
                    mtv = 8893
        elif 3501 <= engine_size <= 4000:
            if price <= 2807700:
                if vehicle_age <= 3:
                    mtv = 129201
                elif 4 <= vehicle_age <= 6:
                    mtv = 111570
                elif 7 <= vehicle_age <= 11:
                    mtv = 65702
                elif 12 <= vehicle_age <= 15:
                    mtv = 29322
                else:
                    mtv = 11674
            else:
                if vehicle_age <= 3:
                    mtv = 140960
                elif 4 <= vehicle_age <= 6:
                    mtv = 121707
                elif 7 <= vehicle_age <= 11:
                    mtv = 71687
                elif 12 <= vehicle_age <= 15:
                    mtv = 31991
                else:
                    mtv = 12734
        elif engine_size > 4000:
            if price <= 3096500:
                if vehicle_age <= 3:
                    mtv = 211479
                elif 4 <= vehicle_age <= 6:
                    mtv = 158577
                elif 7 <= vehicle_age <= 11:
                    mtv = 93917
                elif 12 <= vehicle_age <= 15:
                    mtv = 42208
                else:
                    mtv = 16370
            else:
                if vehicle_age <= 3:
                    mtv = 230698
                elif 4 <= vehicle_age <= 6:
                    mtv = 172998
                elif 7 <= vehicle_age <= 11:
                    mtv = 102458
                elif 12 <= vehicle_age <= 15:
                    mtv = 46044
                else:
                    mtv = 17886
    return mtv
def fill_missing_mtv(df):
    """
    Veri setindeki MTV değerlerini hesaplayarak günceller.

    Args:
        df (pd.DataFrame): İçinde 'EngineSize', 'Year', 'Price(TL)' ve 'VehicleTax()'
                           sütunları bulunan DataFrame.

    Returns:
        pd.DataFrame: MTV değerleri güncellenmiş DataFrame.
    """

    def safe_calculate(row):
        try:
            year = int(row['Year'])
            return calculate_mtv(row['EngineSize'], year, row['Price(TL)'])
        except (ValueError, TypeError):
            return np.nan  # Sayıya çevrilemeyen veya eksik Year için NaN döner

    df['VehicleTax(TL)'] = df.apply(safe_calculate, axis=1)
    return df


# Fonksiyonu kullanarak eksik MTV değerlerini doldur
df = fill_missing_mtv(df)


df.isna().sum()
df.head()
df.info()
df.shape
##############################
# Outlier
##############################
numerical_cols  = [col for col in df.columns if df[col].dtypes not in ['O', 'datetime64[ns]']]

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.98):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit
def check_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in numerical_cols:
        print(col, check_outliers(df, col))


#for col in numerical_cols:
#     sns.histplot(x=df[col].dropna(), kde=True, bins=50)
#     plt.title(f"{col} Distribution")
#     plt.show()
#
#     sk = skew(df[col].dropna())
#     print(f"{col} için çarpıklık (skewness): {sk:.2f}")

def plot_numerical_distributions(df, numerical_cols):
    """
    Verilen sayısal sütunların histogram ve çarpıklık değerlerini subplotlar halinde gösterir.

    Args:
        df (pd.DataFrame): Veri çerçevesi.
        numerical_cols (list): Histogramları çizilecek sayısal sütunların listesi.
        skew_columns (list): skewness değeri -0.5 ile 0.5 arasında olmayan sütunların listesi
    """
    skew_columns = []
    num_cols = len(numerical_cols)
    num_rows = (num_cols + 1) // 2  # Satır sayısını ayarla
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))
    axes = axes.flatten() # 2 boyutlu eksen dizisini tek boyutluya çevir

    for i, col in enumerate(numerical_cols):
        sns.histplot(x=df[col].dropna(), kde=True, bins=50, ax=axes[i])
        axes[i].set_title(f"{col} Dağılımı")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frekans")

        sk = skew(df[col].dropna())
        print(f"{col} için çarpıklık (skewness): {sk:.2f}")

        if sk > 1 or sk < -1:
            skew_columns.append(col)

    # Kullanılmayan subplotları temizle
    if num_cols % 2 != 0:
        fig.delaxes(axes[-1])


    plt.tight_layout()
    plt.show()
    return skew_columns

skew_list = plot_numerical_distributions(df,numerical_cols)

#Eğer skewness değeri:
# > +1 → sağa çarpık (log dönüşüm düşünülebilinir)
# < -1 → sola çarpık (log dönüşüm düşünülebilinir)
# -0.5 ile +0.5 arasında → simetrik sayılır, dönüşüm şart değil

# Log dönüşümüü yapılması gereken değişkenler = skew_list

for col in skew_list:
    df['NEW_'+col+'_log'] = np.log1p(df[col])


# Burada log dönüşümde kullanılan değişkenler veri setinden çıkartılabilinir
# ama hem orijinal hem de log versiyonları karşılaştırmak istediğimiz için ve
# eski değişkenlerden feature extraction yapmak için şimdilik silmiyoruz.
# Log dönüşümü outlier işlemide yapar ancak Outlier baskılaması olmayan değişkenleri outlier baskılama işlemi yapıyoruz.

# sayısal değişkenlerimizi yeniden çağıralım
# yeni oluşturduğumuz log dönüşümlü değerlerimizde de outlier olmuş olabilir.
numerical_cols  = [col for col in df.columns if df[col].dtypes not in ['O', 'datetime64[ns]']]

for col in numerical_cols:
        print(col, check_outliers(df, col))
        if check_outliers(df, col):
            replace_with_threshold(df,col)

df.head()
###########################
# Feature Extraction
##########################

# Address ---> NEW_District
df['NEW_District'] = df['Address'].apply(lambda x: x.split()[-1])
df.drop('Address', axis=1, inplace=True)

# ListingDate ---> NEW_DaysSinceListing
current_date = datetime.now()
df['NEW_DaysSinceListing'] = (current_date - df['ListingDate']).dt.days
# df.drop('ListingDate', axis=1, inplace=True)

# Year ---> NEW_CarAge
df['NEW_CarAge'] = (current_date.date().year - df['Year'])

# PaintAndPartsCondition ---> NEW_NumOriginalParts, NEW_NumLocalPaintedParts, NEW_NumPaintedParts, NEW_NumChangedParts, NEW_NumUnknownParts
parts_list = [
    "Sağ Ön Kapı", "Sağ Arka Kapı", "Sol Ön Kapı", "Sol Arka Kapı",
    "Sağ Ön Çamurluk", "Sol Ön Çamurluk", "Sağ Arka Çamurluk", "Sol Arka Çamurluk",
    "Arka Kaput", "Motor Kaputu", "Tavan", "Ön Tampon", "Arka Tampon"
]
def count_parts_by_category(text):
    categories = ['Orjinal', 'Lokal boyalı', 'Boyalı', 'Değişmiş', 'Belirtilmemiş']
    result = dict.fromkeys([c + "_Count" for c in categories], 0)

    for i, cat in enumerate(categories):
        try:
            start = text.index(cat) + len(cat)
            end = text.index(categories[i + 1]) if i + 1 < len(categories) else len(text)
            segment = text[start:end]
            count = sum(1 for part in parts_list if part in segment)
            result[cat + "_Count"] = count
        except ValueError:
            result[cat + "_Count"] = 0

    return pd.Series(result)
df[['NEW_NumOriginalParts', 'NEW_NumLocalPaintedParts', 'NEW_NumPaintedParts', 'NEW_NumChangedParts', 'NEW_NumUnknownParts']] = df['PaintAndPartsCondition'].apply(count_parts_by_category)
df.drop('PaintAndPartsCondition', axis=1, inplace=True)


# TramerCondition ---> NEW_HasDamage, NEW_HasHighDamage, NEW_DamageCost
def detect_damage(x):
    if pd.isna(x):  # Gerçek NaN
        return -1
    x = str(x).strip().lower().replace('\n', '').replace('\r', '')
    if 'tutarı yok' in x or x == 'yok':
        return 0
    elif 'ağır hasar' in x or any(char.isdigit() for char in x):
        return 1
    elif 'tutarı belirtilmemiş' in x or x == '' or x == 'nan':
        return -1
    else:
        return 1  # Emin olunamayan ama boş olmayan değerler
df['NEW_HasDamage'] = df['TramerCondition'].apply(detect_damage)

df['NEW_HasHighDamage'] = df['TramerCondition'].apply(
    lambda x: 1 if isinstance(x, str) and 'ağır hasar' in x.lower().replace('\n', '').replace('\r', '') else 0)

def extract_cost(val):
    if isinstance(val, str):
        # Sayı içeriyorsa ayıkla
        match = re.search(r'\d[\d\.]*', val)
        if match:
            return float(match.group(0).replace('.', '').replace(',', '.'))
    return np.nan
df['NEW_DamageCost'] = df['TramerCondition'].apply(extract_cost)
df.drop('TramerCondition', axis=1, inplace=True)

# Kilometers --->  NEW_KmPerYear, NEW_KmCategory
df['NEW_KmPerYear'] = df['Kilometers'] / (df['NEW_CarAge'] + 1)

bins = [0, 50000, 100000, 150000, 200000, 300000, np.inf]
labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200-300K", "300K+"]
df["NEW_KmCategory"] = pd.cut(df["Kilometers"], bins=bins, labels=labels)
df["NEW_KmCategory"] = df["NEW_KmCategory"].astype("O")

# Brand + Price(TL) ---> NEW_AvgPricePerBrand
df['NEW_AvgPricePerBrandSeries'] = df.groupby(['Brand', 'Series'])['Price(TL)'].transform('mean')

# Brand + Price(TL) ---> NEW_AvgKmPerBrandSeries
df['NEW_AvgKmPerBrandSeries'] = df.groupby(['Brand', 'Series'])['Kilometers'].transform('mean')


df['NEW_DistrictAvgPrice'] = df.groupby('NEW_District')['Price(TL)'].transform('mean')



#######  Yeni oluşan değişkenlerdeki eksik değerleri doldurma ###########
df.isna().sum()

# NEW_DamageCost
df.loc[df['NEW_HasDamage'] == 1, 'NEW_DamageCost'].fillna(
    df[df['NEW_HasDamage'] == 1].groupby(['Brand', 'Series', 'Year', 'EngineSize'])['NEW_DamageCost'].transform('mean'), inplace=True)

df.loc[(df['NEW_HasDamage'] == 0) & (df['NEW_DamageCost'].isna()), 'NEW_DamageCost'] = 0

df.loc[(df['NEW_HasDamage'] == -1) & (df['NEW_DamageCost'].isna()), 'NEW_DamageCost'] = df['NEW_DamageCost'].mean()

df['NEW_DamageCost'].fillna(df['NEW_DamageCost'].median(), inplace=True)
df.dropna(inplace=True)
########################
# Encode İşlemleri
########################
def rare_encoder(dataframe, column, threshold=0.02):
    value_counts = dataframe[column].value_counts(normalize=True)
    rare_labels = value_counts[value_counts < threshold].index
    dataframe[column] = dataframe[column].apply(lambda x: 'Rare' if x in rare_labels else x)
    return dataframe
def frequency_encoder(dataframe, column):
    freq = dataframe[column].value_counts(normalize=True)
    dataframe[column] = dataframe[column].map(freq)
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# One-hot encode edilecek sütunlar
# GearType(3 sınıf var), FuelType(5 sınıf var),
# DriveType(3 sınıf var), TradeInStatus(3 sınıf var),
# SellerType(4 sınıf var), Brand(10 sınıf var), BodyType(7 sınıf var)

columns_to_oneHotEncode = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() < 11]

df = one_hot_encoder(df, columns_to_oneHotEncode)

# Rare encode edilecek sütunlar
# NEW_District(468 sınıf var), Color(24 sınıf var),
# Series(117 sınıf var), Model(1079 sınıf var)
columns_to_rareEncode = ['Model', 'Series', 'NEW_District', 'Color', 'Brand']
for col in columns_to_rareEncode:
    df = rare_encoder(df,col)


# Frequency encoder uygulanacak değişkenler
# City(80 sınıf var)
# Burada rare encode ettiğimiz sütunları frequency encode işlemi uyguladık bunun yerine one hot encode işlemi de uygulanabilirdi.
columns_to_freqEncode = ['Model', 'Series', 'NEW_District', 'Color', 'Brand', 'City']
for col in columns_to_freqEncode:
    df = frequency_encoder(df,col)

# Burada yola sadece log dönüşümü yaptığımız değişkenler ile devam edebilirz ya da
# dönüşüm yapttığımız değişkenelerin kendilerinide tuttarak model deki etkilerini incelyebiliriz.

#df.drop(['Price(TL)', 'Kilometers', 'EngineSize',  'EnginePower',  'VehicleTax()'], axis=1, inplace=True)

###############################
# Scale İşlemleri
###############################
# Burada kuullanacağımız modellere göre scale işlemi yapılabilir.
# Biz hem ağaç modelleri hemde linear modeller kullanacapımız için veriyi hem scale işlemi olan halini hemde olmayan halini alacağız.


# Scale işleminden önce ListingDate değişeknini çıkartalım !!!!!
df.drop(['ListingDate'], axis=1, inplace=True)

# Log dönşümü, Bool türünde ve Object türünde olan değişkenler hariç diğer değişkenler
columns_to_scale = ['Kilometers', 'EngineSize', 'EnginePower','NEW_DamageCost',
                      'VehicleTax(TL)','NEW_KmPerYear', 'NEW_AvgPricePerBrandSeries',
                      'NEW_AvgKmPerBrandSeries', 'NEW_DistrictAvgPrice']

def scale_features(df, columns_to_scale):
    scaler = StandardScaler()
    df_scaled = df.copy()

    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

    return df_scaled


# Burada Linear model kullanacağımız için Eksik verileri doldurmamız gerekiyor, Karar ağaçları eksik değerler ile çalışabilir.

df_scale = df.copy()
df_scale = scale_features(df_scale, columns_to_scale)



#df.head()
df_scale.head()

# Datayı dışarı aktaralım.
#df.to_csv('dataset/test_data_decision_trees.csv', index=False)
df_scale.to_csv('dataset/test_data_new.csv', index=False)
