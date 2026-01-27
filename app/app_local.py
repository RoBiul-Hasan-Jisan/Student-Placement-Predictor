import streamlit as st
import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f" Model not found! Expected at: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)


st.set_page_config(
    page_title=" Student Placement Predictor",
    layout="centered"
)

st.title(" Student Placement Prediction System")
st.markdown(
    "Predict placement using a **technical skill–weighted LightGBM model**.\n\n"
    "Features prioritized: Maths, Python, SQL > Communication > Mini Projects > Placement Readiness > Attendance"
)

st.divider()


maths = st.slider("Maths Marks", 0, 100, 70)
python = st.slider("Python Marks", 0, 100, 70)
sql = st.slider("SQL Marks", 0, 100, 70)
comm = st.slider("Communication Score", 0, 100, 60)
mini = st.number_input("Mini Projects Completed", 0, 20, 2)
readiness = st.slider("Placement Readiness Score", 0, 100, 65)
attendance = st.slider("Attendance (%)", 0, 100, 75)


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


if st.button(" Predict Placement"):
    probability = model.predict_proba(input_data)[0][1]
    threshold = 0.65

    # Display decision
    if probability >= threshold:
        st.success(f" PLACED (Probability: {probability:.2f})")
    else:
        st.error(f" NOT PLACED (Probability: {probability:.2f})")

    # Progress bar
    st.progress(min(int(probability * 100), 100))

    # Optional: probability gauge / info
    st.info(f"Probability threshold used: {threshold}")
