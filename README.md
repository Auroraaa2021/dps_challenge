# DPS Challenge

## app.py

the deployed model on Google Cloud

use 'curl -X POST -H "Content-Type: application/json" -d "{\"year\": 2020, \"month\": 10}" https://my-flask-app-xx2djygmiq-ez.a.run.app/predict' to test

## dps-challenge.ipynb

data load and pre-process

all the models trained and tested

final model is model2, which is a LSTM with sequence length of 8 trained with total number of all 3 kind of accidents

## result_plot.png

predicted and ground truth of insgesamt Alkoholunfälle value

index is the month number counted from Jan 2000.

## lstm_model.keras & scaler.save

saved model and MinMaxScaler
