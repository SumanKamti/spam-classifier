from flask import Flask, render_template, request, jsonify
import joblib
import os
import math

# -------------------------
# Paths
# -------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "..", "models", "model.pkl")
VECT_PATH  = os.path.join(APP_ROOT, "..", "models", "vectorizer.pkl")

app = Flask(__name__, template_folder="templates")

# -------------------------
# Load model + vectorizer
# -------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# -------------------------
# Helper
# -------------------------
def preprocess(text):
    if text is None:
        return ""
    return str(text).strip().lower()

# -------------------------
# Single-page route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    probability = None

    subject = ""
    email_text = ""

    if request.method == "POST":
        subject = request.form.get("subject", "")
        email_text = request.form.get("email_text", "")

        full_text = (subject + " " + email_text).strip()

        if full_text:
            X = vectorizer.transform([preprocess(full_text)])
            pred = model.predict(X)[0]

            # probability
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(X)[0][1])
            else:
                score = model.decision_function(X)[0]
                probability = 1 / (1 + math.exp(-score))

            label = "SPAM" if int(pred) == 1 else "HAM"

    return render_template(
        "index.html",
        label=label,
        probability=probability,
        subject=subject,
        email_text=email_text
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([preprocess(text)])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        score = model.decision_function(X)[0]
        prob = 1 / (1 + math.exp(-score))

    return jsonify({"label": "spam" if pred==1 else "ham", "probability": prob})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
