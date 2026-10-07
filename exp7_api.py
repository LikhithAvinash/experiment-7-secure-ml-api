import os
import joblib
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "iris_model.pkl"

# Read API key from environment variable
# Default value is only for local testing
API_KEY = os.environ.get("API_KEY", "secret123")


# Train and save model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    X, y = load_iris(return_X_y=True)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)


# Load saved model
model = joblib.load(MODEL_PATH)

# Iris class names
TARGETS = load_iris().target_names


app = Flask(__name__)


def authenticate(req):
    """Check whether the API key is valid."""

    if req.headers.get("x-api-key") != API_KEY:
        return {"error": "Unauthorized - invalid API key"}, 401

    return None


def validate(payload):
    """Validate the incoming JSON request."""

    if not payload or "features" not in payload:
        return {"error": "Missing 'features' in request body"}, 400

    features = payload["features"]

    if not isinstance(features, list) or len(features) != 4:
        return {"error": "'features' must be a list of 4 numbers"}, 400

    if not all(isinstance(v, (int, float)) for v in features):
        return {"error": "All features must be numeric"}, 400

    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "iris-rf"
    })


@app.route("/predict", methods=["POST"])
def predict():

    # Authenticate before accessing the model
    auth_error = authenticate(request)

    if auth_error:
        return jsonify(auth_error[0]), auth_error[1]

    # Read JSON body
    payload = request.get_json(silent=True)

    # Validate input
    val_error = validate(payload)

    if val_error:
        return jsonify(val_error[0]), val_error[1]

    # Make prediction
    pred = int(model.predict([payload["features"]])[0])

    # Get prediction probabilities
    proba = model.predict_proba([payload["features"]])[0]

    return jsonify({
        "prediction": pred,
        "class": str(TARGETS[pred]),
        "confidence": round(float(max(proba)), 3)
    })


if __name__ == "__main__":

    client = app.test_client()

    print("=== Secure Flask Prediction API ===")

    # 1. Health check
    r = client.get("/health")
    print(
        f"GET  /health              -> "
        f"{r.status_code} {r.get_json()}"
    )

    # 2. No API key
    r = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )

    print(
        f"POST /predict (no key)    -> "
        f"{r.status_code} {r.get_json()}"
    )

    # 3. Wrong API key
    r = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers={"x-api-key": "wrong"}
    )

    print(
        f"POST /predict (bad key)   -> "
        f"{r.status_code} {r.get_json()}"
    )

    # 4. Bad input
    r = client.post(
        "/predict",
        json={"features": [1.4]},
        headers={"x-api-key": API_KEY}
    )

    print(
        f"POST /predict (bad input) -> "
        f"{r.status_code} {r.get_json()}"
    )

    # 5. Valid request
    r = client.post(
        "/predict",
        json={"features": [6.7, 3.0, 5.2, 2.3]},
        headers={"x-api-key": API_KEY}
    )

    print(
        f"POST /predict (valid)     -> "
        f"{r.status_code} {r.get_json()}"
    )