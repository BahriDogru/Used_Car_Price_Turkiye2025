
# Used Car Price Prediction Project 🚗💰

This project is a machine learning pipeline to predict second-hand car prices based on various features scraped from online car listings in Turkey.

## 📌 Project Purpose

The main objective of this project is to predict the `Price(TRY)` of a car using features such as brand, model, year, mileage, and other attributes. The project involves data preprocessing, feature engineering, handling missing values, encoding categorical variables, detecting and treating outliers, model training and evaluation, and finally, deploying the model via a Streamlit app.

## 📁 Dataset

The dataset was compiled by scraping publicly available used car listings from a popular car market in Turkey. It contains extensive details about vehicles, sellers and their geographical locations. After extensive cleaning and pre-processing, the dataset is structured and ready to be used for vehicle price prediction and machine learning tasks. The dataset presented here is in its raw form without any data pre-processing. For the cleaned version of the dataset, you can use the Preprocessing_pipeline.py file or you can find it on my kaggle page.


👉 **[Download Dataset](https://your-dataset-link-here.com)**

## 🧩 Features (Variables)

| Variable Name             | Description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| `Title`                   | Title of the car listing                                                 |
| `Address`                 | Neighborhood/Street information from the listing                         |
| `City`                    | City where the vehicle is listed                                         |
| `Price(TRY)`              | Vehicle price in Turkish Lira (Target variable)                          |
| `ListingID`               | Unique identifier for the listing                                        |
| `ListingDate`             | Date when the vehicle was listed                                         |
| `Brand`                   | Brand of the vehicle (e.g., Toyota, BMW, Fiat)                           |
| `Series`                  | Series name under the brand (e.g., Corolla, Golf)                        |
| `Model`                   | Full model name of the vehicle                                           |
| `Year`                    | Model year of the car                                                    |
| `Kilometers`              | Mileage of the vehicle in kilometers                                     |
| `GearType`                | Transmission type (Manual, Automatic, Semi-automatic)                    |
| `FuelType`                | Fuel type used (Gasoline, Diesel, LPG, Hybrid, Electric)                 |
| `BodyType`                | Body structure of the car (Sedan, Hatchback, SUV, etc.)                  |
| `Color`                   | Exterior color of the vehicle                                            |
| `EngineSize`              | Engine displacement (in cc)                                              |
| `EnginePower`             | Engine power (in horsepower)                                             |
| `DriveType`               | Traction type (e.g., Front-Wheel Drive, Rear-Wheel Drive, 4WD)           |
| `PaintAndPartsCondition`  | Status of paint and parts (original, painted, changed, etc.)             |
| `TradeInStatus`           | Whether the seller accepts trade-in vehicles                             |
| `SellerType`              | Seller type (Private seller, Dealer, Authorized Dealer, Unspecified)     |
| `VehicleTax(TRY)`         | Annual motor vehicle tax (MTV) amount                                    |
| `TramerCondition`         | Damage/accident history or insurance record from the Tramer system       |

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
git clone https://github.com/BahriDogru/Used_Car_Price_Turkiye2025.git

# Navigate into the folder
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```
## 🚀 Streamlit Web Application

To make the project interactive and more accessible, we deployed it as a web application using **Streamlit**. This allows users to input vehicle details through a user-friendly interface and instantly receive a predicted car price based on our trained machine learning model.

The app not only provides a prediction but also includes useful visualizations and market analysis to help users better understand the pricing dynamics.

⚠️ **Disclaimer**:  
This is a prototype model trained on a specific dataset and may not reflect real-world market conditions perfectly. The predictions are for informational purposes only and should not be considered as definitive pricing advice. Further improvements and more comprehensive data would be needed for commercial use.

🌐 **Try the app here**: [Car Price Prediction App](https://carvalue-genie.streamlit.app)


## 🙌 Acknowledgements

This project was developed by a team of four passionate data enthusiasts:

- Enes GÜLER [@enesgulerml](https://github.com/enesgulerml)
- Beyaz KARAYILAN [@beyazss](https://github.com/beyazss)
- Hatice Rüveyda AKÇA [@ruveydaruby](https://github.com/ruveydaruby)
- Bahhri DOĞRU [@BahriDogru](https://github.com/BahriDogru)

Thanks to all contributors!

---