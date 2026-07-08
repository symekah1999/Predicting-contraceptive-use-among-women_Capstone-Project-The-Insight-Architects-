"""
app.py — Flask deployment API for the KDHS 2022 Contraceptive Use model.

    GET  /health         -> liveness check
    POST /predict         -> single-respondent prediction
    POST /predict_batch   -> list of respondent records
"""
from flask import Flask, request, jsonify
import joblib
import traceback

from feature_engineering import engineer_features

app = Flask(__name__)

bundle = joblib.load('contraceptive_model_bundle.joblib')
model = bundle['model']
preprocessor = bundle['preprocessor']
label_encoder = bundle['label_encoder']

REQUIRED_FIELDS = [
    'age', 'age_group', 'county', 'residence_type', 'education_level', 'religion',
    'household_size', 'household_head_sex', 'wealth_index', 'children_ever_born',
    'age_first_birth', 'living_children', 'pregnancy_loss', 'marital_status',
    'union_status', 'partner_education', 'currently_working',
]


def _predict_one(record: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f'Missing required field(s): {missing}')

    row = engineer_features(record)
    X = preprocessor.transform(row)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = label_encoder.inverse_transform([pred])[0]
    proba_by_class = {cls: round(float(p), 4) for cls, p in zip(label_encoder.classes_, proba)}
    return {
        'prediction': label,
        'probabilities': proba_by_class,
        'risk_flag': 'non_use_risk' if label == 'No method' else 'using_method',
    }


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'LightGBM (KDHS 2022 contraceptive-use classifier)'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = _predict_one(request.get_json(force=True))
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        return jsonify({'error': 'internal_error', 'trace': traceback.format_exc()}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        records = request.get_json(force=True)
        if not isinstance(records, list):
            return jsonify({'error': 'Expected a JSON list of respondent records'}), 400
        results = [_predict_one(r) for r in records]
        return jsonify({'count': len(results), 'results': results}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        return jsonify({'error': 'internal_error', 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
