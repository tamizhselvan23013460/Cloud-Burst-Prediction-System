from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# ============================================================
# 🌩️ LOAD MODEL, SCALER, AND THRESHOLD
# ============================================================
model = joblib.load("cloudburst_model.pkl")   # stacking or xgboost model
scaler = joblib.load("scaler_stacking.pkl")

try:
    with open("best_threshold_stacking.txt", "r") as f:
        best_threshold = float(f.read().strip())
except:
    best_threshold = 0.5

# ============================================================
# 🏠 HOME ROUTE
# ============================================================
@app.route('/')
def home():
    return render_template('cloud_burst.html')

# ============================================================
# 🔮 PREDICTION API ROUTE (for fetch)
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch input fields (names must match your HTML)
        rainfall = float(request.form['rainfall'])
        avg_temp = float(request.form['temperature'])
        evaporation = float(request.form['evaporation'])
        avg_humidity = float(request.form['humidity'])
        wind_gust_speed = float(request.form['wind'])
        avg_pressure = float(request.form['pressure'])

        # Derived features (auto backend)
        temp_range = 0
        wind_gust_speed_sq = wind_gust_speed ** 2
        pressure_drop_index = 0
        saturation_deficit = avg_temp * (100 - avg_humidity)
        rainfall_wind_interaction = rainfall * wind_gust_speed

        # Create feature vector
        features = np.array([[
            avg_temp, rainfall, evaporation, avg_humidity,
            wind_gust_speed, avg_pressure, temp_range,
            wind_gust_speed_sq, pressure_drop_index,
            saturation_deficit, rainfall_wind_interaction
        ]])

        # Scale and predict
        scaled_features = scaler.transform(features)
        prob = model.predict_proba(scaled_features)[0, 1]
        prediction = "Yes" if prob > best_threshold else "No"

        return jsonify({
            "prediction": prediction,
            "probability": round(float(prob), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ============================================================
# 🚀 RUN FLASK APP
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
