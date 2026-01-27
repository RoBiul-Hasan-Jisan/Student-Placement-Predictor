from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # Get form data
        maths = float(request.form.get("maths", 70))
        python = float(request.form.get("python", 70))
        sql = float(request.form.get("sql", 70))
        comm = float(request.form.get("comm", 60))
        mini = float(request.form.get("mini", 2))
        readiness = float(request.form.get("readiness", 65))
        attendance = float(request.form.get("attendance", 75))

        # Feature Engineering
        Maths_w  = maths * 2.0
        Python_w = python * 2.0
        SQL_w    = sql * 2.0
        Comm_w   = comm * 1.5
        Mini_w   = np.log1p(mini)
        Ready_w  = readiness * 0.7
        Attend_w = attendance * 0.5

        input_data = np.array([[
            Maths_w, Python_w, SQL_w,
            Comm_w, Mini_w,
            Ready_w, Attend_w
        ]])

        # Prediction
        probability = model.predict_proba(input_data)[0][1]
        threshold = 0.65
        prediction = "PLACED" if probability >= threshold else "NOT PLACED"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
