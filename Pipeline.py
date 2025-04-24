import joblib
import pandas as pd
import numpy as np
import re
import warnings
from warnings import filterwarnings
filterwarnings('ignore')
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from scipy.stats import skew
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Data Preprocessing & Feature Engineering
def get_col_names(dataframe, categorical_th = 10, cardinal_th = 20):

    # Cardinal and Categorical Columns
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numerical_but_categorical_columns = [col for col in dataframe.columns if dataframe[col].nunique() < categorical_th and
                                         dataframe[col].dtypes != "O"]
    categorical_but_cardinal_columns = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_th and
                                        dataframe[col].dtypes == "O"]
    categorical_columns = categorical_columns + numerical_but_categorical_columns
    categorical_columns = [col for col in categorical_columns if col not in categorical_but_cardinal_columns]

    # Numerical Columns
    numerical_columns = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    numerical_columns = [col for col in dataframe.columns if col not in numerical_but_categorical_columns]

    print(f'Categorical Columns: {len(categorical_columns)}')
    print(f'Numerical Columns: {len(numerical_columns)}')
    print(f'Categorical But Cardinal Columns: {len(categorical_but_cardinal_columns)}')
    print(f'Numerical But Cardinal Columns: {len(numerical_but_categorical_columns)}')
    return categorical_columns, numerical_columns, categorical_but_cardinal_columns

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    q1 = dataframe[variable].quantile(q1)
    q3 = dataframe[variable].quantile(q3)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def replace_with_thresholds(dataframe,variable):
    lower_bound, upper_bound = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < lower_bound, variable] = lower_bound
    dataframe.loc[dataframe[variable] > upper_bound, variable] = upper_bound

def extract_number_safe(x):
        if pd.isnull(x):
            return np.nan
        num = re.sub(r"[^\d]", "", str(x))
        return int(num) if num else np.nan

def calculate_mtv(engine_size, vehicle_year, price):
        current_year = datetime.now().year
        try:
            vehicle_age = current_year - int(vehicle_year)
            engine_size = float(engine_size)
            price = float(price)
        except ValueError:
            return np.nan
        mtv = 0

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
        df['VEHICLETAX'] = df.apply(lambda row: calculate_mtv(row['ENGINESIZE'], row['YEAR'], row['PRICE']), axis=1)
        return df

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

def detect_damage(x):
        if pd.isna(x):
            return -1
        x = str(x).strip().lower().replace('\n', '').replace('\r', '')
        if 'tutarı yok' in x or x == 'yok':
            return 0
        elif 'ağır hasar' in x or any(char.isdigit() for char in x):
            return 1
        elif 'tutarı belirtilmemiş' in x or x == '' or x == 'nan':
            return -1
        else:
            return 1

def extract_cost(val):
        if isinstance(val, str):
            match = re.search(r'\d[\d\.]*', val)
            if match:
                return float(match.group(0).replace('.', '').replace(',', '.'))
        return np.nan

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

def scale_features(df, columns_to_scale):
        scaler = StandardScaler()
        df_scaled = df.copy()

        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

        return df_scaled

def all_data_preprocessing(df):
    df.columns = df.columns.str.strip()
    df.columns = [col.upper() for col in df.columns]
    df.columns = [col.upper().replace("(TL)", "").strip() for col in df.columns]
    df.columns = [col.upper().replace("()", "").strip() for col in df.columns]

    df["PRICE"] = df["PRICE"].apply(extract_number_safe)
    df["KILOMETERS"] = df["KILOMETERS"].apply(extract_number_safe)
    df["VEHICLETAX"] = df["VEHICLETAX"].apply(extract_number_safe)
    df["ENGINESIZE"] = df["ENGINESIZE"].apply(extract_number_safe)
    df["ENGINEPOWER"] = df["ENGINEPOWER"].apply(extract_number_safe)

    df['SELLERTYPE'] = df['SELLERTYPE'].str.strip()
    df["SELLERTYPE"] = df["SELLERTYPE"].apply(lambda x: "Belirtilmemiş" if str(x).strip() == "" else x)

    df['TRADEINSTATUS'] = df['TRADEINSTATUS'].str.strip()
    df["TRADEINSTATUS"] = df["TRADEINSTATUS"].apply(lambda x: "Belirtilmemiş" if str(x).strip() == "" else x)
    df = df[df["TRADEINSTATUS"].isin(["Takasa Uygun", "Takasa Uygun Değil", "Belirtilmemiş"])]

    df["DRIVETYPE"] = df["DRIVETYPE"].str.strip()
    df = df[df["DRIVETYPE"].isin(["Önden Çekiş", "Arkadan İtiş", "4WD (Sürekli)", "AWD (Elektronik)"])]

    df["COLOR"] = df["COLOR"].str.strip()
    df = df[~df["COLOR"].isin(["Cabrio", "Coupe"])]

    df["BODYTYPE"] = df["BODYTYPE"].str.strip()
    df = df[~df["BODYTYPE"].str.contains("^\s*$")]
    df = df[~df["BODYTYPE"].isin(["-", " "])]

    df["FUELTYPE"] = df["FUELTYPE"].str.strip()

    df['GEARTYPE'] = df['GEARTYPE'].str.strip()

    month_map = {
        "Ocak": "January", "Şubat": "February", "Mart": "March", "Nisan": "April",
        "Mayıs": "May", "Haziran": "June", "Temmuz": "July", "Ağustos": "August",
        "Eylül": "September", "Ekim": "October", "Kasım": "November", "Aralık": "December"}

    for tr, en in month_map.items():
        df['LISTINGDATE'] = df['LISTINGDATE'].str.replace(tr, en)

    df['LISTINGDATE'] = df['LISTINGDATE'].str.strip()
    df['LISTINGDATE'] = df['LISTINGDATE'].str.replace(r"\s+", " ", regex=True)
    df['LISTINGDATE'] = pd.to_datetime(df['LISTINGDATE'], format='%d %B %Y')

    ### Outlier ###
    lower_price, upper_price = outlier_thresholds(df, "PRICE")
    lower_vt, upper_vt = outlier_thresholds(df, "VEHICLETAX")

    replace_with_thresholds(df, "PRICE")
    replace_with_thresholds(df, "VEHICLETAX")

    ### Missing Values ###

    df["SELLERTYPE"].fillna("Sahibinden", inplace=True)
    df.dropna(subset=["ENGINESIZE"], inplace=True)
    df = fill_missing_mtv(df)

    df['NEW_District'] = df['ADDRESS'].apply(lambda x: x.split()[-1])
    df.drop('ADDRESS', axis=1, inplace=True)

    current_date = datetime.now()
    df['NEW_DaysSinceListing'] = (current_date - df['LISTINGDATE']).dt.days

    df['NEW_CarAge'] = (current_date.date().year - df['YEAR']).astype(int)

    df[['NEW_NumOriginalParts', 'NEW_NumLocalPaintedParts', 'NEW_NumPaintedParts', 'NEW_NumChangedParts',
        'NEW_NumUnknownParts']] = df['PAINTANDPARTSCONDITION'].apply(count_parts_by_category)
    df.drop('PAINTANDPARTSCONDITION', axis=1, inplace=True)

    df['NEW_HasDamage'] = df['TRAMERCONDITION'].apply(detect_damage)

    df['NEW_HasHighDamage'] = df['TRAMERCONDITION'].apply(
        lambda x: 1 if isinstance(x, str) and 'ağır hasar' in x.lower().replace('\n', '').replace('\r', '') else 0)

    df['NEW_DamageCost'] = df['TRAMERCONDITION'].apply(extract_cost)
    df.drop('TRAMERCONDITION', axis=1, inplace=True)

    df['NEW_KmPerYear'] = df['KILOMETERS'] / (df['NEW_CarAge'] + 1)

    bins = [0, 50000, 100000, 150000, 200000, 300000, np.inf]
    labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200-300K", "300K+"]
    df["NEW_KmCategory"] = pd.cut(df["KILOMETERS"], bins=bins, labels=labels)
    df["NEW_KmCategory"] = df["NEW_KmCategory"].astype("O")

    df['NEW_AvgPricePerBrandSeries'] = df.groupby(['BRAND', 'SERIES'])['PRICE'].transform('mean')

    df['NEW_AvgKmPerBrandSeries'] = df.groupby(['BRAND', 'SERIES'])['KILOMETERS'].transform('mean')

    df['NEW_DistrictAvgPrice'] = df.groupby('NEW_District')['PRICE'].transform('mean')

    df.loc[df['NEW_HasDamage'] == 1, 'NEW_DamageCost'].fillna(
        df[df['NEW_HasDamage'] == 1].groupby(['BRAND', 'SERIES', 'YEAR', 'ENGINESIZE'])['NEW_DamageCost'].transform(
            'mean'), inplace=True)

    df.loc[(df['NEW_HasDamage'] == 0) & (df['NEW_DamageCost'].isna()), 'NEW_DamageCost'] = 0

    df.loc[(df['NEW_HasDamage'] == -1) & (df['NEW_DamageCost'].isna()), 'NEW_DamageCost'] = df['NEW_DamageCost'].mean()

    df['NEW_DamageCost'].fillna(df['NEW_DamageCost'].median(), inplace=True)

    ## Encoding ##
    columns_to_oneHotEncode = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() < 11]
    df = one_hot_encoder(df, columns_to_oneHotEncode)

    columns_to_rareEncode = ['MODEL', 'SERIES', 'NEW_District', 'COLOR']

    for col in columns_to_rareEncode:
        df = rare_encoder(df, col)

    columns_to_freqEncode = ['MODEL', 'SERIES', 'NEW_District', 'COLOR', 'CITY']

    for col in columns_to_freqEncode:
        df = frequency_encoder(df, col)

    df.drop(['LISTINGDATE'], axis=1, inplace=True)

    columns_to_scale = ['PRICE', 'KILOMETERS', 'ENGINESIZE', 'ENGINEPOWER', 'NEW_DamageCost',
                        'VEHICLETAX', 'NEW_KmPerYear', 'NEW_AvgPricePerBrandSeries',
                        'NEW_AvgKmPerBrandSeries', 'NEW_DistrictAvgPrice']

    df_scale = df.copy()
    df_scale = scale_features(df_scale, columns_to_scale)

    df = df.drop(["TITLE", "LISTINGID"], axis=1)

    X = df.drop(["PRICE"], axis=1)
    y = df["PRICE"]

    return X, y

# Data Preprocessing & Feature Engineering

# Base Models
def base_models(X, y, scoring="r2"):
    print("Modelling Base Models...")
    regression = [('KNN', KNeighborsRegressor()),
                   ("CART", DecisionTreeRegressor(random_state = 42)),
                   ("RF", RandomForestRegressor(random_state = 42)),
                   ('Adaboost', AdaBoostRegressor(random_state = 42)),
                   ('GBM', GradientBoostingRegressor(random_state = 42)),
                   ('XGBoost', XGBRegressor(use_label_encoder=False,random_state = 42, eval_metric='logloss')),
                   ('LightGBM', LGBMRegressor(verbose=-1, random_state = 42))
                   ]
    for name, regressor in regression:
        cv_results = cross_validate(regressor, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
# Base Models

# Hyperparameter Optimization
knn_params = {"n_neighbors": range(2,350)}

cart_params = {'max_depth': [3,5,7,9,None],
               "min_samples_split": range(2,30),
               "min_samples_leaf": [1,2,4,6,8,10],
               "max_features": [None,"sqrt","log2"],
               "max_leaf_nodes": [None,10,20,30,40,50],
               "ccp_alpha": [0.0,0.01,0.05,0.1]}

rf_params = {"max_depth": [3,5,7,9,None],
             "max_features": [5,7,"sqrt","log2",None],
             "min_samples_split": [2,4,6,8,10,11],
             "n_estimators": [200,500,1000,2000]}

gbm_params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 4, 5, 6],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.6, 0.8, 1.0],
    "max_features": ["auto", "sqrt", "log2"]
}

xgboost_params = {"learning_rate": [0.3,0.1,0.05,0.01],
                  "max_depth": [3,5,8],
                  "n_estimators": [200,500,1000,2000],
                  "colsample_bytree": [0.5,1]}

lightgbm_params = {"learning_rate": [0.3,0.1,0.05,0.01],
                   "n_estimators": [200,500,1000,2000],
                   "colsample_bytree": [0.5,0.7,1]}

regressor = [('KNN', KNeighborsRegressor(),knn_params),
               ("CART", DecisionTreeRegressor(random_state = 42), cart_params),
               ("RF", RandomForestRegressor(random_state = 42), rf_params),
               ("GBM", GradientBoostingRegressor(random_state = 42), gbm_params),
               ('XGBoost', XGBRegressor(eval_metric='rmse', random_state=42), xgboost_params),
               ('LightGBM', LGBMRegressor(verbose=-1,random_state = 42), lightgbm_params)]
# Hyperparameter Optimization

def hyperparameter_optimization(X, y, cv=5, scoring="r2"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, rsor, params in regressor:
        print(f"######## {name} Model is Running ########")
        cv_results = cross_validate(rsor, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(rsor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = rsor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
        print(f"######## {name} Model Finished ########")
    return best_models

# Stacking & Ensemble Learning
def voting_regression(best_models, X, y):
    print("######## Voting Regressor is Running ########")
    voting_reg = VotingRegressor(
        estimators=
        [('KNN', best_models["KNN"]),
         ('RF', best_models["RF"]),
         ('LightGBM', best_models["LightGBM"])])

    cv_results = cross_validate(voting_reg, X, y, cv=5, scoring=["neg_mean_squared_error", "r2", "neg_mean_absolute_error"])
    print(f"MSE: {-cv_results['test_neg_mean_squared_error'].mean()}")
    print(f"MAE: {-cv_results['test_neg_mean_absolute_error'].mean()}")
    print(f"R2 : {cv_results['test_r2'].mean()}")
    print("######## Voting Regressor Finished ########")
    return voting_reg
# Stacking & Ensemble Learning

# Main #
def main():
    df = pd.read_csv("car_price.csv")
    X, y = all_data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_models(X_train, y_train)
    best_models = hyperparameter_optimization(X_train, y_train)
    voting_clf = voting_regression(best_models, X_train, y_train)
    voting_clf.fit(X_train,y_train)
    y_pred = voting_clf.predict(X_test)
    print("\n--- Final Test Set Performance ---")
    print(f"R2   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("Modeling has begun...")
    main()
# Main #