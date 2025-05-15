from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "weather_secret_key"

# Load models and preprocessors
aus_preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
aus_lstm = load_model("lstm_model.h5")
aus_ann = load_model("ann_model.h5")
aus_svm = pickle.load(open("svm_model.pkl", "rb"))

tnj_scaler = pickle.load(open("thanjavur_preprocessor_scaler.pkl", "rb"))
tnj_lstm = load_model("thanjavur_lstm_model.h5")
tnj_ann = load_model("thanjavur_ann_model.h5")
tnj_svm = pickle.load(open("thanjavur_svm_model.pkl", "rb"))

# Features
numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

aus_locations = ["Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat", "Bendigo", "Brisbane", "Cairns", "Canberra",
                 "Cobar", "CoffsHarbour", "Dartmoor", "Darwin", "GoldCoast", "Hobart", "Katherine", "Launceston", "Melbourne",
                 "MelbourneAirport", "Mildura", "Moree", "MountGambier", "MountGinini", "Newcastle", "Nhil", "NorahHead",
                 "NorfolkIsland", "Nuriootpa", "PearceRAAF", "Penrith", "Perth", "PerthAirport", "Portland", "Richmond", "Sale",
                 "SalmonGums", "Sydney", "SydneyAirport", "Townsville", "Tuggeranong", "Uluru", "WaggaWagga", "Walpole", "Watsonia",
                 "Williamtown", "Witchcliffe", "Wollongong", "Woomera"]

wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW',
                   'NNW', 'NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW']

rain_today_options = ['Yes', 'No']

thanjavur_features = ['Temperature', 'MaxTemp', 'MinTemp', 'Humidity', 'Rainfall', 'Pressure', 'WindSpeed']

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

# 🏠 Front page: Choose dataset
@app.route('/')
def home():
    return render_template("select.html")

# Handle dataset choice and redirect
@app.route('/choose', methods=['POST'])
def choose_dataset():
    dataset = request.form.get('dataset', 'Australia')
    session['dataset'] = dataset
    return redirect(url_for('input_form'))

# Show input form based on dataset
@app.route('/input')
def input_form():
    dataset = session.get('dataset', 'Australia')
    return render_template("index.html",
                           dataset=dataset,
                           aus_features=numeric_features,
                           thanjavur_features=thanjavur_features,
                           aus_locations=aus_locations,
                           wind_directions=wind_directions,
                           rain_today_options=rain_today_options,
                           prediction=None)

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    dataset = session.get('dataset', 'Australia')
    prediction_text = "No Rain Tomorrow..."

    if dataset == "Australia":
        numeric_inputs = [safe_float(request.form.get(f, '0')) for f in numeric_features]
        raw_categorical = {
            'Location': request.form.get('Location', ''),
            'WindGustDir': request.form.get('WindGustDir', ''),
            'WindDir9am': request.form.get('WindDir9am', ''),
            'WindDir3pm': request.form.get('WindDir3pm', ''),
            'RainToday': request.form.get('RainToday', 'No')
        }

        input_data = {**dict(zip(numeric_features, numeric_inputs)), **raw_categorical}
        input_df = pd.DataFrame([input_data])
        transformed = aus_preprocessor.transform(input_df)
        lstm_input = transformed.reshape((1, 1, transformed.shape[1]))

        lstm_out = aus_lstm.predict(lstm_input, verbose=0)
        ann_out = aus_ann.predict(lstm_out, verbose=0)
        prediction = aus_svm.predict(ann_out)[0]
        if prediction == 1:
            prediction_text = "Yes, It will Rain Tomorrow!!"

    else:
        tnj_inputs = [safe_float(request.form.get(f, '0')) for f in thanjavur_features]
        raintoday_binary = 1 if request.form.get('RainToday', 'No') == 'Yes' else 0
        final_input = tnj_inputs + [raintoday_binary]
        scaled_input = tnj_scaler.transform([final_input])
        lstm_input = scaled_input.reshape((1, 1, len(final_input)))
        lstm_out = tnj_lstm.predict(lstm_input, verbose=0)
        ann_out = tnj_ann.predict(lstm_out, verbose=0)
        prediction = tnj_svm.predict(ann_out)[0]
        if prediction == 1:
            prediction_text = "Yes, It will Rain Tomorrow!!"

    return render_template("index.html",
                           dataset=dataset,
                           aus_features=numeric_features,
                           thanjavur_features=thanjavur_features,
                           aus_locations=aus_locations,
                           wind_directions=wind_directions,
                           rain_today_options=rain_today_options,
                           prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
