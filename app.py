"""
app.py — Flask REST API for the KDHS 2022 Contraceptive Use model
==================================================================

Endpoints
---------
GET  /               HTML form front-end (for non-technical users)
GET  /health         Liveness check — no auth required
POST /predict        Single respondent → prediction + probabilities
POST /predict_batch  JSON list of respondents → list of predictions

Authentication
--------------
Every POST request must include the header:
    X-API-Key: <your key>

Set the key before starting:
    Windows Git Bash : export API_KEY="your-secret-key"
    Linux / macOS    : export API_KEY="your-secret-key"
    Docker           : docker run -e API_KEY="your-secret-key" ...

If API_KEY is not set, the server auto-generates one at startup and
prints it to the console once — copy it from there.

Running the server
------------------
Development:
    python app.py

Production (gunicorn):
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 60 app:app
"""

import os
import secrets
import traceback

from flask import Flask, request, jsonify, render_template
import joblib

from feature_engineering import engineer_features, REQUIRED_FIELDS

# ── Initialise app ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model bundle once at startup ─────────────────────────────────────
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), 'contraceptive_model_bundle.joblib')
bundle        = joblib.load(BUNDLE_PATH)
model         = bundle['model']
preprocessor  = bundle['preprocessor']
label_encoder = bundle['label_encoder']

print(f"Model loaded: {BUNDLE_PATH}")
print(f"Classes: {list(label_encoder.classes_)}")

# ── API key setup ──────────────────────────────────────────────────────────
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    API_KEY = secrets.token_urlsafe(24)
    print("=" * 70)
    print("No API_KEY environment variable set — generated one for this run:")
    print(f"  API_KEY = {API_KEY}")
    print("Pass this as the X-API-Key header on every POST request.")
    print("To keep a fixed key across restarts:  export API_KEY=<your-key>")
    print("=" * 70)


# ── Helper functions ───────────────────────────────────────────────────────
def _check_auth() -> bool:
    """Return True if the X-API-Key header is present and correct."""
    supplied = request.headers.get('X-API-Key', '')
    return supplied == API_KEY


def _predict_one(record: dict) -> dict:
    """
    Validate one respondent record, run feature engineering + inference,
    and return a prediction dict.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    row   = engineer_features(record)
    X     = preprocessor.transform(row)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label          = label_encoder.inverse_transform([pred])[0]
    proba_by_class = {
        cls: round(float(p), 4)
        for cls, p in zip(label_encoder.classes_, proba)
    }

    return {
        'prediction':   label,
        'probabilities': proba_by_class,
        'risk_flag':    'non_use_risk' if label == 'No method' else 'using_method',
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    """
    Serve the HTML form so non-technical users (CHPs, county officers)
    can get predictions without writing any code.
    """
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Liveness check — no authentication required."""
    return jsonify({
        'status': 'ok',
        'model':  'LightGBM (KDHS 2022 — Contraceptive Use Classifier)',
        'classes': list(label_encoder.classes_),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the contraceptive-use category for a single respondent.

    Request body (JSON):
        {
          "age": 28, "age_group": "25-29", "county": "Turkana",
          "residence_type": "Rural", "education_level": "Primary",
          "religion": "Roman Catholic", "household_size": 6,
          "household_head_sex": "Male", "wealth_index": "Poorest",
          "children_ever_born": 3, "age_first_birth": 19,
          "living_children": 3, "pregnancy_loss": "No",
          "marital_status": "Married", "union_status": "Currently in union",
          "partner_education": "Primary", "currently_working": "No"
        }

    Response (JSON):
        {
          "prediction": "No method",
          "probabilities": {"Folkloric": 0.0003, "Modern": 0.406,
                            "No method": 0.522, "Traditional": 0.0717},
          "risk_flag": "non_use_risk"
        }
    """
    if not _check_auth():
        return jsonify({'error': 'Unauthorized — missing or invalid X-API-Key header'}), 401
    try:
        record = request.get_json(force=True)
        return jsonify(_predict_one(record)), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        return jsonify({'error': 'internal_error', 'trace': traceback.format_exc()}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for a list of respondent records in one call.

    Request body: JSON array of respondent dicts (same fields as /predict).
    Response: { "count": N, "results": [ ... ] }
    """
    if not _check_auth():
        return jsonify({'error': 'Unauthorized — missing or invalid X-API-Key header'}), 401
    try:
        records = request.get_json(force=True)
        if not isinstance(records, list):
            return jsonify({'error': 'Expected a JSON array of respondent records'}), 400
        results = [_predict_one(r) for r in records]
        return jsonify({'count': len(results), 'results': results}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        return jsonify({'error': 'internal_error', 'trace': traceback.format_exc()}), 500


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
