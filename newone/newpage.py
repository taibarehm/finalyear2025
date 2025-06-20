from flask import Flask, jsonify, render_template, request
import subprocess
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
import os
app = Flask(__name__)

# Load the CatBoost model once at startup
model = joblib.load('catboost_stress_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/boot.html')
def boot():
    return render_template('boot.html')
@app.route('/save-chat', methods=['POST'])
def save_chat():
    data = request.json['chat']
    df = pd.DataFrame(data)
    # Trim whitespace from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Add current date and time columns
    now = datetime.now()
    df['date'] = now.strftime('%Y-%m-%d')
    df['time'] = now.strftime('%H:%M:%S')
    file_exists = os.path.isfile('chat_data.xlsx')
    with pd.ExcelWriter('chat_data.xlsx', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=not file_exists, sheet_name='ChatData')
    return jsonify({'status': 'success'})
@app.route('/stress-detect', methods=['POST'])
def stress_detect():
    # Example: get form data from POST request
    try:
        features = [
            float(request.form['snoring_range']),
            float(request.form['respiration_rate']),
            float(request.form['body_temperature']),
            float(request.form['limb_movement']),
            float(request.form['blood_oxygen']),
            float(request.form['eye_movement']),
            float(request.form['hours_of_sleep']),
            float(request.form['heart_rate'])
        ]
        pred = model.predict([features])[0]
        message = f"Predicted Stress Level: {int(pred)}"
    except Exception as e:
        message = f"Error: {e}"
    return render_template('home.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)