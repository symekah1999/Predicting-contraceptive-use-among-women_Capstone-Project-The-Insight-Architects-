import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the full serialized pipeline (contains preprocessor + classifier)
pipeline_path = os.path.join(os.path.dirname(__file__), 'contraceptive_pipeline.joblib')

try:
    pipeline = joblib.load(pipeline_path)
    print("Successfully loaded model pipeline from:", pipeline_path)
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'status': 'error', 'message': 'Model pipeline not loaded on server.'}), 500
        
    try:
        # Check if the request is JSON or form-encoded
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        # Parse and type-cast incoming values
        age = int(data.get('age', 25))
        household_size = int(data.get('household_size', 4))
        children_ever_born = int(data.get('children_ever_born', 0))
        living_children = int(data.get('living_children', 0))
        age_first_birth = float(data.get('age_first_birth', 0))
        has_given_birth = 1 if children_ever_born > 0 else 0
        if children_ever_born == 0:
            age_first_birth = 0.0

        # Construct single-row DataFrame matching training columns exactly
        input_row = pd.DataFrame([{
            'age': age,
            'county': data.get('county', 'Nairobi'),
            'residence_type': data.get('residence_type', 'Urban'),
            'education_level': data.get('education_level', 'Secondary'),
            'religion': data.get('religion', 'Protestant/Other Christian'),
            'household_size': household_size,
            'household_head_sex': data.get('household_head_sex', 'Male'),
            'wealth_index': data.get('wealth_index', 'Middle'),
            'children_ever_born': children_ever_born,
            'age_first_birth': age_first_birth,
            'living_children': living_children,
            'pregnancy_loss': data.get('pregnancy_loss', 'No'),
            'marital_status': data.get('marital_status', 'Married'),
            'union_status': data.get('union_status', 'Currently in union'),
            'partner_education': data.get('partner_education', 'Secondary'),
            'currently_working': data.get('currently_working', 'Yes'),
            'has_given_birth': has_given_birth
        }])

        # Generate probabilities and predictions
        prob = pipeline.predict_proba(input_row)[0][1]
        pred = int(pipeline.predict(input_row)[0])

        return jsonify({
            'status': 'success',
            'prediction': 'Modern Contraceptive User' if pred == 1 else 'Non-user / Traditional / Folkloric',
            'probability': round(float(prob) * 100, 2)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }), 400

if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
