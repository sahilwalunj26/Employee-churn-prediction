import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "naive_bayes.pkl"

app = Flask(__name__)

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

# IMPORTANT: These mappings are common for the Employee Future Prediction dataset.
# Confirm them against the encoders/mappings used when you trained your model.
ENCODINGS = {
    "education": {"Bachelors": 0, "Masters": 1, "PHD": 2},
    "city": {"City 1": 0, "City 2": 1, "City 3": 2},
    "gender": {"Female": 0, "Male": 1},
    "ever_benched": {"No": 0, "Yes": 1},
}

FEATURES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
]


def to_int(form, field, label, minimum, maximum):
    raw = form.get(field, "").strip()
    if not raw:
        raise ValueError(f"Please enter {label.lower()}.")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number.")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            education = request.form.get("education", "")
            city = request.form.get("city", "")
            gender = request.form.get("gender", "")
            ever_benched = request.form.get("ever_benched", "")

            for field, value, mapping, label in [
                ("education", education, ENCODINGS["education"], "Education"),
                ("city", city, ENCODINGS["city"], "City"),
                ("gender", gender, ENCODINGS["gender"], "Gender"),
                ("ever_benched", ever_benched, ENCODINGS["ever_benched"], "Ever benched"),
            ]:
                if value not in mapping:
                    raise ValueError(f"Please select a valid {label.lower()} option.")

            joining_year = to_int(request.form, "joining_year", "Joining year", 2000, 2100)
            payment_tier = to_int(request.form, "payment_tier", "Payment tier", 1, 3)
            age = to_int(request.form, "age", "Age", 18, 100)
            experience = to_int(request.form, "experience", "Experience in current domain", 0, 50)

            row = [[
                ENCODINGS["education"][education],
                joining_year,
                ENCODINGS["city"][city],
                payment_tier,
                age,
                ENCODINGS["gender"][gender],
                ENCODINGS["ever_benched"][ever_benched],
                experience,
            ]]

            # Preserve the feature names/order expected by the fitted model.
            prediction_input = row
            if hasattr(model, "feature_names_in_"):
                import pandas as pd
                prediction_input = pd.DataFrame(row, columns=FEATURES)

            prediction = int(model.predict(prediction_input)[0])
            if prediction == 1:
                result = {
                    "class": prediction,
                    "title": "Likely to leave",
                    "description": "The model classified this employee as class 1.",
                    "status": "warning",
                }
            else:
                result = {
                    "class": prediction,
                    "title": "Likely to stay",
                    "description": "The model classified this employee as class 0.",
                    "status": "success",
                }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction could not be completed. Check that the model and input encodings match the training setup."

    return render_template("index.html", result=result, error=error, form_data=form_data)


@app.route("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
