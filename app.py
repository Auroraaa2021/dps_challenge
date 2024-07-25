from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load the model and the scaler
model = load_model('lstm_model.keras')
scaler = joblib.load('scaler.save')

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = int(data.get('year'))
    month = int(data.get('month'))

    # Convert year and month to MonthNumber
    base_year = 2000
    base_month = 1
    month_number = (year - base_year) * 12 + (month - base_month) + 1

    # Generate the historical data needed to create the sequence
    input_df = pd.read_csv('historical_data.csv')

    sequence_length = 8
    input_df = input_df.iloc[:month_number]
    scaled_input = scaler.transform(input_df[['Alkoholunfälle', 'Verkehrsunfälle', 'Fluchtunfälle']])


    input_sequence, _ = create_sequences(scaled_input[-sequence_length - 1:], sequence_length)
    #print(input_sequence.shape)

    prediction_scaled = model.predict(input_sequence)

    # Make a prediction
    prediction_scaled = model.predict(input_sequence)
    #print(prediction_scaled.shape)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]
    #print(prediction)

    return jsonify({"prediction": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
    
#curl -X POST -H "Content-Type: application/json" -d "{\"year\": 2020, \"month\": 10}" https://lstm-predictor-xx2djygmiq-ey.a.run.app/predict
#curl -X POST -H "Content-Type: application/json" -d @test_payload.json https://YOUR_CLOUD_RUN_URL/predict
