import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "naive_bayes.pkl"

app = Flask(__name__)

# Keep naive_bayes.pkl in the same directory as this app.py file.
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

# IMPORTANT: Verify these encodings against the exact mappings used during training.
ENCODINGS = {
    "education": {"Bachelors": 0, "Masters": 1, "PHD": 2},
    "city": {"City 1": 0, "City 2": 1, "City 3": 2},
    "gender": {"Female": 0, "Male": 1},
    "ever_benched": {"No": 0, "Yes": 1},
}

FEATURES = [
    "Education", "JoiningYear", "City", "PaymentTier",
    "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain"
]

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Employee Insight | Prediction</title>
  <style>
    :root { --ink:#182230; --muted:#667085; --line:#e4e9f1; --brand:#625bf6; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
      background:#f4f6fb; color:var(--ink); }
    .topbar { background:#111827; color:white; padding:18px 24px; }
    .top-inner { max-width:1000px; margin:auto; display:flex; align-items:center; gap:12px; }
    .logo { width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
      background:linear-gradient(135deg,#7c73ff,#5145cd); font-size:22px; }
    .brand { font-weight:750; letter-spacing:.2px; }
    .subbrand { color:#c6c9d5; font-size:12px; margin-top:3px; }
    main { max-width:1000px; margin:38px auto; padding:0 18px 40px; }
    .eyebrow { color:var(--brand); font-size:12px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
    h1 { font-size:clamp(28px,4vw,39px); letter-spacing:-1px; margin:10px 0 9px; }
    .intro { color:var(--muted); line-height:1.6; margin:0 0 25px; }
    .layout { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(250px,.8fr); gap:20px; align-items:start; }
    .panel { background:white; border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:0 8px 28px #17203308; }
    .panel h2 { font-size:18px; margin:0 0 6px; }
    .hint { color:var(--muted); font-size:13px; margin:0 0 22px; line-height:1.5; }
    .fields { display:grid; grid-template-columns:1fr 1fr; gap:17px; }
    label { display:block; font-size:13px; font-weight:650; margin-bottom:8px; }
    input,select { width:100%; min-height:45px; border:1px solid #d7deea; border-radius:10px; padding:10px 12px;
      background:#fff; color:#202939; font:inherit; outline:none; }
    input:focus,select:focus { border-color:var(--brand); box-shadow:0 0 0 3px #625bf61b; }
    .submit { width:100%; margin-top:22px; min-height:48px; border:0; border-radius:11px; color:white;
      background:var(--brand); font-size:15px; font-weight:750; cursor:pointer; }
    .submit:hover { background:#5148db; }
    .result { border-radius:13px; padding:16px; margin-bottom:18px; }
    .success { background:#ecfdf3; border:1px solid #abefc6; }
    .warning { background:#fff5ed; border:1px solid #fed7aa; }
    .result small { color:#475467; font-weight:750; text-transform:uppercase; letter-spacing:.7px; }
    .result h3 { margin:8px 0 6px; font-size:21px; }
    .result p { margin:0; color:#475467; line-height:1.5; font-size:13px; }
    .error { padding:12px 14px; border-radius:10px; background:#fef3f2; color:#b42318; margin-bottom:16px; font-size:14px; }
    .info-list { display:grid; gap:14px; margin-top:20px; }
    .info-item { display:flex; gap:11px; align-items:flex-start; }
    .icon { width:34px; height:34px; display:grid; place-items:center; flex:0 0 34px; border-radius:10px; background:#f0efff; }
    .info-item strong { display:block; font-size:13px; margin:1px 0 4px; }
    .info-item span { display:block; font-size:12px; color:var(--muted); line-height:1.5; }
    .note { margin-top:18px; font-size:11px; color:#7b8494; line-height:1.5; }
    footer { text-align:center; color:#8790a0; font-size:12px; padding:0 18px 25px; }
    @media(max-width:760px) { .layout { grid-template-columns:1fr; } main { margin-top:27px; } }
    @media(max-width:480px) { .fields { grid-template-columns:1fr; } .panel { padding:18px; } .topbar { padding:15px 18px; } }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="top-inner">
      <div class="logo">✦</div>
      <div><div class="brand">Employee Insight</div><div class="subbrand">Gaussian Naive Bayes · ML Demo</div></div>
    </div>
  </header>
  <main>
    <div class="eyebrow">Machine learning · Classification</div>
    <h1>Employee Retention Prediction</h1>
    <p class="intro">Enter the employee details below to get a prediction from your trained Naive Bayes model.</p>
    <div class="layout">
      <section class="panel">
        <h2>Employee details</h2>
        <p class="hint">Complete all fields. Choose categorical values from the dropdown menus.</p>
        {% if error %}<div class="error" role="alert">{{ error }}</div>{% endif %}
        {% if result %}
          <div class="result {{ result.status }}" role="status">
            <small>Prediction result · Class {{ result.class_id }}</small>
            <h3>{{ result.title }}</h3>
            <p>{{ result.description }}</p>
          </div>
        {% endif %}
        <form method="post">
          <div class="fields">
            <div><label for="education">Education</label>
              <select id="education" name="education" required>
                {% for value in ["Bachelors","Masters","PHD"] %}<option value="{{ value }}" {% if form_data.get('education','Bachelors') == value %}selected{% endif %}>{{ value }}</option>{% endfor %}
              </select>
            </div>
            <div><label for="joining_year">Joining year</label>
              <input id="joining_year" name="joining_year" type="number" min="2000" max="2100" placeholder="e.g. 2018" value="{{ form_data.get('joining_year','') }}" required>
            </div>
            <div><label for="city">City</label>
              <select id="city" name="city" required>
                {% for value in ["City 1","City 2","City 3"] %}<option value="{{ value }}" {% if form_data.get('city','City 1') == value %}selected{% endif %}>{{ value }}</option>{% endfor %}
              </select>
            </div>
            <div><label for="payment_tier">Payment tier</label>
              <select id="payment_tier" name="payment_tier" required>
                {% for value in [1,2,3] %}<option value="{{ value }}" {% if form_data.get('payment_tier','2')|string == value|string %}selected{% endif %}>Tier {{ value }}</option>{% endfor %}
              </select>
            </div>
            <div><label for="age">Age</label>
              <input id="age" name="age" type="number" min="18" max="100" placeholder="e.g. 28" value="{{ form_data.get('age','') }}" required>
            </div>
            <div><label for="gender">Gender</label>
              <select id="gender" name="gender" required>
                {% for value in ["Female","Male"] %}<option value="{{ value }}" {% if form_data.get('gender','Male') == value %}selected{% endif %}>{{ value }}</option>{% endfor %}
              </select>
            </div>
            <div><label for="ever_benched">Ever benched?</label>
              <select id="ever_benched" name="ever_benched" required>
                {% for value in ["No","Yes"] %}<option value="{{ value }}" {% if form_data.get('ever_benched','No') == value %}selected{% endif %}>{{ value }}</option>{% endfor %}
              </select>
            </div>
            <div><label for="experience">Experience in current domain (years)</label>
              <input id="experience" name="experience" type="number" min="0" max="50" placeholder="e.g. 3" value="{{ form_data.get('experience','') }}" required>
            </div>
          </div>
          <button class="submit" type="submit">Predict employee outcome&nbsp; →</button>
        </form>
      </section>
      <aside class="panel">
        <h2>About this model</h2>
        <p class="hint">A machine learning demo using your saved classification model.</p>
        <div class="info-list">
          <div class="info-item"><div class="icon">🧠</div><div><strong>Gaussian Naive Bayes</strong><span>Uses the trained model stored in your PKL file.</span></div></div>
          <div class="info-item"><div class="icon">📋</div><div><strong>8 input features</strong><span>Education, joining year, city, payment tier, age, gender, bench status and domain experience.</span></div></div>
          <div class="info-item"><div class="icon">⚡</div><div><strong>Instant prediction</strong><span>Submit the form to see the predicted class and its configured label.</span></div></div>
        </div>
        <p class="note">Demo only. Verify categorical encodings and target-class meanings against your training code before using results for real employment decisions.</p>
      </aside>
    </div>
  </main>
  <footer>Employee Insight · Machine Learning Demo</footer>
</body>
</html>
"""

def to_int(form, field, label, minimum, maximum):
    raw = form.get(field, "").strip()
    if not raw:
        raise ValueError(f"Please enter {label.lower()}.")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number.") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            selected = {
                "education": request.form.get("education", ""),
                "city": request.form.get("city", ""),
                "gender": request.form.get("gender", ""),
                "ever_benched": request.form.get("ever_benched", ""),
            }
            for field, value in selected.items():
                if value not in ENCODINGS[field]:
                    raise ValueError(f"Please select a valid {field.replace('_', ' ')} option.")

            row = [[
                ENCODINGS["education"][selected["education"]],
                to_int(request.form, "joining_year", "Joining year", 2000, 2100),
                ENCODINGS["city"][selected["city"]],
                to_int(request.form, "payment_tier", "Payment tier", 1, 3),
                to_int(request.form, "age", "Age", 18, 100),
                ENCODINGS["gender"][selected["gender"]],
                ENCODINGS["ever_benched"][selected["ever_benched"]],
                to_int(request.form, "experience", "Experience in current domain", 0, 50),
            ]]
            prediction_input = pd.DataFrame(row, columns=FEATURES)
            prediction = int(model.predict(prediction_input)[0])

            if prediction == 1:
                result = {
                    "class_id": prediction,
                    "title": "Likely to leave",
                    "description": "The model classified this employee as class 1.",
                    "status": "warning",
                }
            else:
                result = {
                    "class_id": prediction,
                    "title": "Likely to stay",
                    "description": "The model classified this employee as class 0.",
                    "status": "success",
                }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Check the model file, scikit-learn version, and training encodings."

    return render_template_string(PAGE, result=result, error=error, form_data=form_data)


@app.route("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
