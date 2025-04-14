import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings


warnings.filterwarnings('ignore')
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)

df = pd.read_csv("dataset/car_price.csv")
df.info()
df.head()

# Tüm sütunlardaki string değerlerin başındaki ve sonundaki boşlukları temizle
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# remove Title and ListteningID
drop_list = ['Title', 'ListingID']
df = df.drop(drop_list, axis=1)

######################
## Dtype düzeltme
######################

# 'Price(TL)' sütununu temizle ve float'a çevir
df['Price(TL)'] = df['Price(TL)'].str.replace('.', '', regex=False)  # binlik ayıracı olan noktaları kaldır
df['Price(TL)'] = df['Price(TL)'].str.replace(' TL', '', regex=False)  # ' TL' kısmını kaldır
df['Price(TL)'] = df['Price(TL)'].str.replace(',', '.', regex=False)  # eğer ondalık virgül varsa noktaya çevir
df['Price(TL)'] = pd.to_numeric(df['Price(TL)'], errors='coerce')



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

###########################
# Edit Variables - Missing Values
##########################

# BodyType
list_BodyType = ['Hatchback/5', 'Sedan', 'Coupe', 'Hatchback/3','Roadster','Station wagon','MPV']
df = df[df["BodyType"].isin(list_BodyType)]

# DriveType
list_DriveType = ['Önden Çekiş', '4WD (Sürekli)','Arkadan İtiş']
df = df[df["DriveType"].isin(list_DriveType)]

# FuelType
list_FuelType = ['Benzin', 'Dizel', 'Hibrit', 'LPG & Benzin','Elektrik']
df = df[df["FuelType"].isin(list_FuelType)]


#  TradeInStatus, SellerType missing values
# print(df['TradeInStatus'].value_counts(dropna=False))
# print(df['SellerType'].value_counts(dropna=False))

df['TradeInStatus'].replace('', np.nan, inplace=True)
df['TradeInStatus'].fillna('Belirtilmemiş', inplace=True)


df['SellerType'].replace('', np.nan, inplace=True)
df['SellerType'].fillna('Belirtilmemiş', inplace=True)



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
    df['VehicleTax()'] = df.apply(lambda row: calculate_mtv(row['EngineSize'], row['Year'], row['Price(TL)']), axis=1)
    return df

# Fonksiyonu kullanarak eksik MTV değerlerini doldur
df = fill_missing_mtv(df)

df.head()
df.info()

###########################
# Feature Extraction
##########################

# Address ---> Feature Extraction
df['District'] = df['Address'].str.extract(r'Mh\.\s*(.*)')
df.drop('Address', axis=1, inplace=True)


# ListingDate ---> Feature Extraction
current_date = datetime.now()
df['DaysSinceListing'] = (current_date - df['ListingDate']).dt.days
# df.drop('ListingDate', axis=1, inplace=True)


# PaintAndPartsCondition
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
df[['NumOriginalParts', 'NumLocalPaintedParts', 'NumPaintedParts', 'NumChangedParts', 'NumUnknownParts']] = df['PaintAndPartsCondition'].apply(count_parts_by_category)
df.drop('PaintAndPartsCondition', axis=1, inplace=True)



# TramerCondition
df['TramerCondition'] = df['TramerCondition'].astype(str).str.strip()

def detect_damage(x):
    if pd.isna(x):  # Gerçek NaN
        return np.nan
    x = str(x).strip().lower()
    if x == 'tutarı yok':
        return 0
    elif 'ağır hasar' in x or any(char.isdigit() for char in x):
        return 1
    elif 'tutarı belirtilmemiş' in x or x == '' or x == 'nan':
        return np.nan
    else:
        return 1  # Emin olunamayan ama boş olmayan değerler
df['HasDamage'] = df['TramerCondition'].apply(detect_damage)

df['HasHighDamage'] = df['TramerCondition'].apply(
    lambda x: 1 if isinstance(x, str) and 'ağır hasar' in x.lower() else 0)

def extract_cost(val):
    if isinstance(val, str):
        # Sayı içeriyorsa ayıkla
        match = re.search(r'\d[\d\.]*', val)
        if match:
            return float(match.group(0).replace('.', '').replace(',', '.'))
    return np.nan
df['DamageCost'] = df['TramerCondition'].apply(extract_cost)
df.drop('TramerCondition', axis=1, inplace=True)

df['Year'] = df['Year'].astype(int)




##############################
# Outlier
##############################
numerical_cols  = ['DamageCost','VehicleTax()', 'Kilometers', 'Year', 'Price(TL)']

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
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


for col in numerical_cols:
    sns.histplot(x=df[col].dropna(), kde=True, bins=50)
    plt.title(f"{col} Distribution")
    plt.show()

    sk = skew(df[col].dropna())
    print(f"{col} için çarpıklık (skewness): {sk:.2f}")
def plot_numerical_distributions(df, numerical_cols):
    """
    Verilen sayısal sütunların histogram ve çarpıklık değerlerini subplotlar halinde gösterir.

    Args:
        df (pd.DataFrame): Veri çerçevesi.
        numerical_cols (list): Histogramları çizilecek sayısal sütunların listesi.
    """
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

    # Kullanılmayan subplotları temizle
    if num_cols % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()
plot_numerical_distributions(df,numerical_cols)

#Eğer skewness değeri:
# > +1 → sağa çarpık (log dönüşüm düşünülebilinir)
# < -1 → sola çarpık (log dönüşüm düşünülebilinir)
# -0.5 ile +0.5 arasında → simetrik sayılır, dönüşüm şart değil

# Log dönüşümüü yapılması gereken değişkenler = 'DamageCost','VehicleTax()', 'Kilometers', 'Price(TL)'

df['DamageCost_log'] = np.log1p(df['DamageCost'])
df['VehicleTax_log'] = np.log1p(df['VehicleTax()'])
df['Kilometers_log'] = np.log1p(df['Kilometers'])
df['Price_log'] = np.log1p(df['Price(TL)'])

# Burada log dönüşümde kullanılan değişkenler veri setinden çıkartılabilinir
# ama hem orijinal hem de log versiyonları karşılaştırmak istediğimiz için şimdilik silmiyoruz.
# df.drop(['DamageCost', 'VehicleTax(TL)', 'Kilometers', 'Price(TL)'], axis=1, inplace=True)

numerical_cols2 = ['DamageCost_log', 'VehicleTax_log', 'Kilometers_log', 'Price_log']

for col in numerical_cols2:
        print(col, check_outliers(df, col))

replace_with_threshold(df,'Kilometers_log')



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

columns_to_oneHotEncode = ['Brand','GearType', 'FuelType', 'DriveType',
                     'TradeInStatus', 'SellerType',  'BodyType', 'TradeInStatus', 'SellerType']

df = one_hot_encoder(df, columns_to_oneHotEncode)


# Rare encode edilecek sütunlar
# District(468 sınıf var), Color(31 sınıf var),
# Series(123 sınıf var), Model(1139 sınıf var)
columns_to_rareEncode = ['Model', 'Series', 'District', 'Color']
for col in columns_to_rareEncode:
    df = rare_encoder(df,col)


# Frequency encoder uygulanacak değişkenler
# City(82 sınıf var)
# Burada rare encode ettiğimiz sütunları frequency encode işlemi uyguladık bunun yerine one hot encode işlemi de uygulanabilirdi.
columns_to_freqEncode = ['Model', 'Series', 'District', 'Color', 'City']
for col in columns_to_freqEncode:
    df = frequency_encoder(df,col)

