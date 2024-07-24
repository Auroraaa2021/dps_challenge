from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

# Load the model and the scaler
model = load_model('lstm_model.keras')
scaler = joblib.load('scaler.save')

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data['year']
    month = data['month']

    # Convert year and month to MonthNumber
    base_year = 2000
    base_month = 1
    month_number = (year - base_year) * 12 + (month - base_month) + 1

    # Generate the historical data needed to create the sequence
    input_df = pd.read_csv('historical_data.csv')
    
    # Normalize the data
    scaled_input = scaler.transform(input_df[['Alkoholunfälle', 'Verkehrsunfälle', 'Fluchtunfälle']])

    # Create the input sequence for the LSTM model
    sequence_length = 8
    if len(scaled_input) < sequence_length:
        return jsonify({'error': 'Not enough data to create a sequence'})

    input_sequence = np.array([scaled_input[-sequence_length:]])

    # Make a prediction
    prediction_scaled = model.predict(input_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)[0]

    return jsonify({
        'prediction': prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)