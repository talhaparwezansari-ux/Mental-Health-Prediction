
import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "Mansik_santulant.pkl"))

# Country groups
top_countries = [
    'Other', 'India', 'USA', 'Canada', 'Australia',
    'UK', 'Germany', 'Mexico', 'Turkey', 'France'
]


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data received'}), 400

        # Group country
        country_group = (
            data['country']
            if data['country'] in top_countries
            else 'Other'
        )

        # Prepare input for ML model
        input_data = pd.DataFrame([{
            'Study_Hours': data['study_hours'],
            'Age': data['age'],
            'Avg_Daily_Usage_Hours': data['avg_daily_usage_hour'],
            'Daily_Unlocks': data['daily_unlocks'],
            'Physical_Activity_Hours': data['physical_activity_hours'],
            'Sleep_Hours_Per_Night': data['sleep_hours_per_night'],
            'Stress_Level': data['stress_level'],
            'Gender': data['gender'],
            'Academic_Level': data['academic_level'],
            'Most_Used_Platform': data['most_used_platform'],
            'Purpose_Of_Use': data['purpose_of_use'],
            'grouped_country': country_group
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return jsonify({
            'predicted_mental_health_score': round(float(prediction), 2)
        })

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
