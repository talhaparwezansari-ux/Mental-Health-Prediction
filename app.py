import joblib
import pandas as pd

from flask import Flask, request, jsonify ,render_template
from flask_cors import CORS


# Load trained model
model = joblib.load('Mansik_santulant.pkl')


app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})


# Allowed values
top_countries = [
    'Other', 'India', 'USA', 'Canada', 'Australia', 'UK',
    'Germany', 'Mexico', 'Turkey', 'France'
]

allowed_values = {
    'gender': ['Male', 'Female'],

    'academic_level': [
        'Undergraduate', 'Graduate', 'High School'
    ],

    'most_used_platform': [
        'Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
        'Twitter', 'YouTube', 'TikTok', 'LINE',
        'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat'
    ],

    'purpose_of_use': [
        'Networking', 'Education', 'Entertainment', 'News'
    ],

    'stress_level': [
        'Medium', 'Low', 'Very High', 'High'
    ]
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400


        # Validate required fields
        required_fields = [
            'age',
            'gender',
            'country',
            'academic_level',
            'most_used_platform',
            'purpose_of_use',
            'avg_daily_usage_hour',
            'daily_unlocks',
            'study_hours',
            'physical_activity_hours',
            'sleep_hours_per_night',
            'stress_level'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400


        # Validate categorical columns
        for field, values in allowed_values.items():

            if data[field] not in values:
                return jsonify({
                    'error': f'Invalid {field}. Allowed values: {values}'
                }), 400


        # Validate numerical columns
        age = data['age']
        avg_daily_usage_hour = data['avg_daily_usage_hour']
        daily_unlocks = data['daily_unlocks']
        study_hours = data['study_hours']
        physical_activity_hours = data['physical_activity_hours']
        sleep_hours_per_night = data['sleep_hours_per_night']


        if not isinstance(age, int) or isinstance(age, bool) or not 10 <= age <= 100:
            return jsonify({
                'error': 'Age must be an integer between 10 and 100'
            }), 400


        if not isinstance(avg_daily_usage_hour, (int, float)) or isinstance(avg_daily_usage_hour, bool) or not 0 <= avg_daily_usage_hour <= 24:
            return jsonify({
                'error': 'Average daily usage must be between 0 and 24'
            }), 400


        if not isinstance(daily_unlocks, int) or isinstance(daily_unlocks, bool) or daily_unlocks < 0:
            return jsonify({
                'error': 'Daily unlocks must be a non-negative integer'
            }), 400


        if not isinstance(study_hours, int) or isinstance(study_hours, bool) or not 0 <= study_hours <= 24:
            return jsonify({
                'error': 'Study hours must be an integer between 0 and 24'
            }), 400


        if not isinstance(physical_activity_hours, int) or isinstance(physical_activity_hours, bool) or not 0 <= physical_activity_hours <= 24:
            return jsonify({
                'error': 'Physical activity hours must be an integer between 0 and 24'
            }), 400


        if not isinstance(sleep_hours_per_night, (int, float)) or isinstance(sleep_hours_per_night, bool) or not 0 <= sleep_hours_per_night <= 24:
            return jsonify({
                'error': 'Sleep hours must be between 0 and 24'
            }), 400


        # Group country
        country_group = (
            data['country']
            if data['country'] in top_countries
            else 'Other'
        )


        # Create input DataFrame
        input_row = pd.DataFrame([{

            'Age': data['age'],

            'Gender': data['gender'],

            'Country': data['country'],

            'Academic_Level': data['academic_level'],

            'Most_Used_Platform': data['most_used_platform'],

            'Purpose_Of_Use': data['purpose_of_use'],

            'Avg_Daily_Usage_Hours': data['avg_daily_usage_hour'],

            'Daily_Unlocks': data['daily_unlocks'],

            'Study_Hours': data['study_hours'],

            'Physical_Activity_Hours': data['physical_activity_hours'],

            'Sleep_Hours_Per_Night': data['sleep_hours_per_night'],

            'Stress_Level': data['stress_level'],

            'grouped_country': country_group

        }])


        prediction = model.predict(input_row)[0]

        print("Predicted Score:", prediction)

        return jsonify({
            'predicted_mental_health_score': round(float(prediction), 2)
        })


    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# Run Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
