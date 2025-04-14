# Used Car Price Prediction Project 🚗💰

This project is a machine learning pipeline to predict second-hand car prices based on various features scraped from online car listings in Turkey.

## 📌 Project Purpose

The main objective of this project is to predict the `Price(TL)` of a car using features such as brand, model, year, mileage, and other attributes. The project involves data preprocessing, feature engineering, handling missing values, encoding categorical variables, detecting and treating outliers, model training and evaluation, and finally, deploying the model via a Streamlit app.

## 📁 Dataset

The dataset was created by scraping car listings and contains various details about the vehicles, sellers, and locations. The dataset can be downloaded from the following link:

👉 **[Download Dataset](https://your-dataset-link-here.com)**

## 🧩 Features (Variables)

| Variable Name     | Description                        |
|-------------------|------------------------------------|
| `Title`           | Ad Title                           |
| `Brand`           | Car brand                          |
| `Model`           | Car model                          |
| `Series`          | Model series                       |
| `BodyType`        | Body type of the vehicle           |
| `FuelType`        | Type of fuel                       |
| `GearType`        | Gearbox type                       |
| `DriveType`       | Drive system (e.g., FWD, RWD, AWD) |
| `Year`            | Model year                         |
| `Kilometers`      | Total distance traveled (in km)    |
| `Color`           | Exterior color                     |
| `City`            | City where the car is listed       |
| `District`        | District of the listing            |
| `DamageCost`      | Estimated damage cost              |
| `VehicleTax`      | Annual vehicle tax                 |
| `SellerType`      | Type of seller (individual/dealer) |
| `TradeInStatus`   | Whether trade-in is accepted       |
| `Price(TL)`       | Listing price (target variable)    |

## 🧪 What Has Been Done

- ✅ Web scraping from second-hand car websites
- ✅ Data cleaning and preprocessing
- ✅ Handling missing values
- ✅ Encoding categorical variables (One-Hot, Frequency, Rare encoding)
- ✅ Log transformation of skewed features
- ✅ Outlier detection and capping
- ✅ Model training using various regressors (DecisionTree, RandomForest, etc.)
- ✅ Model evaluation and comparison
- ✅ Streamlit app for deployment

## 🛠️ How to Use

```bash
# Clone the repo
git clone https://github.com/BahriDogru/Used-Car-Listings---Turkey-2025.git

# Navigate into the folder
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## 🙌 Acknowledgements

This project was developed by a team of four passionate data enthusiasts:

- Enes GÜLER [@enesgulerml](https://github.com/enesgulerml)
- Beyaz KARAYILAN [@beyazss](https://github.com/beyazss)
- Hatice Rüveyda AKÇA [@ruveydaruby](https://github.com/ruveydaruby)
- Bahhri DOĞRU [@BahriDogru](https://github.com/BahriDogru)

Thanks to all contributors!

---