from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def predict_feature(data_df, selected_month, input_year):
    data_df["JAN-MAR"] = data_df[["JAN", "FEB", "MAR"]].mean(axis=1)
    data_df["APR-JUN"] = data_df[["APR", "MAY", "JUN"]].mean(axis=1)
    data_df["JUL-SEP"] = data_df[["JUL", "AUG", "SEP"]].mean(axis=1)
    data_df["OCT-DEC"] = data_df[["OCT", "NOV", "DEC"]].mean(axis=1)
    
    x = data_df['YEAR'].to_numpy()
    y = data_df[selected_month].to_numpy()
    
    model = ARIMA(y, order=(5,1,0))
    model_fit = model.fit()
    
    future_years = np.arange(x[-1] + 1, input_year + 1)
    prediction = model_fit.forecast(steps=len(future_years))
    return prediction[-1]

@app.route("/", methods=["GET"])
def home():
    return "Backend is working!"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        lat = float(data['lat'])
        lon = float(data['lon'])
        month = data['month']
        year = int(data['year'])

        # --- Temperature ---
        temp_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M&community=SB&longitude={lon}&latitude={lat}&start=1981&end=2023&format=CSV"
        temp_df = pd.read_csv(temp_url, skiprows=9)
        temperature = predict_feature(temp_df, month, year)

        # --- Humidity ---
        humi_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=RH2M&community=SB&longitude={lon}&latitude={lat}&start=1981&end=2023&format=CSV"
        humi_df = pd.read_csv(humi_url, skiprows=9)
        humidity = predict_feature(humi_df, month, year)

        # --- Rainfall ---
        rain_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR&community=SB&longitude={lon}&latitude={lat}&start=1981&end=2023&format=CSV"
        rain_df = pd.read_csv(rain_url, skiprows=9)
        
        # Convert to mm
        rain_df[["JAN","MAR","MAY","JUL","AUG","OCT","DEC"]] *= 31
        rain_df[["APR","JUN","SEP","NOV"]] *= 30
        rain_df["FEB"] *= 29 if is_leap_year(year) else 28
        rainfall = predict_feature(rain_df, month, year)

        # --- Crop prediction ---
        crop_df = pd.read_csv("Crop_recommendation.csv")
        crop_df.drop(['N', 'P', 'K', 'ph'], axis=1, inplace=True)
        X = crop_df[['temperature', 'humidity', 'rainfall']].values
        le = LabelEncoder()
        y = le.fit_transform(crop_df['label'].values)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        input_scaled = scaler.transform([[temperature, humidity, rainfall]])

        models = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=100, random_state=42),
            DecisionTreeClassifier(random_state=42),
            KNeighborsClassifier(n_neighbors=5),
            SVC(kernel='rbf', probability=True)
        ]

        crop_set = set()
        for model in models:
            model.fit(X, y)
            pred = model.predict(input_scaled)
            crop_set.add(le.inverse_transform([pred[0]])[0])

        return jsonify({
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "rainfall": round(rainfall, 2),
            "predicted_crops": list(crop_set)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

