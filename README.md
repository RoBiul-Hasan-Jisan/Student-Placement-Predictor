#  Student Placement Predictor

[![Model: LightGBM](https://img.shields.io/badge/Model-LightGBM-blue.svg)](https://lightgbm.readthedocs.io/)
[![Accuracy: 99.5%](https://img.shields.io/badge/Accuracy-99.5%25-green.svg)](#)
[![Interface: Flask & Gradio](https://img.shields.io/badge/UI-Flask%20%7C%20Gradio-orange.svg)](#)

A high-precision predictive system designed to determine student placement probability. The model utilizes a weighted feature engineering approach, prioritizing core technical skills and placement readiness.

---

##  Project Overview
This project goes beyond simple classification by applying **domain-specific weights** to features. It mimics real-world recruitment criteria where technical proficiency (Python, SQL, Maths) is weighted more heavily than secondary factors like attendance.

###  Key Highlights
* **Weighted Logic:** Custom transformation of raw scores based on industry importance.
* **Dual UI:** Includes both a **Flask Web App** (Production-style) and a **Gradio Interface** (Rapid prototyping).
* **High Precision:** Achieves **~99.5% accuracy** using the LightGBM algorithm.
* **Tunable Threshold:** Uses a custom decision threshold of `0.65` to ensure conservative and reliable placement predictions.

---

##  Feature Engineering & Weights
To improve prediction accuracy, raw inputs are transformed using the following weighting system:

| Feature | Weight / Transformation | Priority |
| :--- | :--- | :--- |
| **Applied Maths** | $\times 2.0$ | High |
| **Python** | $\times 2.0$ | High |
| **SQL** | $\times 2.0$ | High |
| **Communication Score** | $\times 1.5$ | Medium |
| **Mini Projects** | $\log(1+x) \times 1.0$ | Low-Medium |
| **Placement Readiness** | $\times 0.7$ | Medium |
| **Attendance** | $\times 0.5$ | Low |



---

##  Model Performance
The model was built using the **LightGBM Classifier**, optimized for speed and handling complex feature relationships.

* **Test Accuracy:** 99.5%
* **ROC-AUC:** 1.0
* **Decision Threshold:** 0.65 (Placed if $P > 0.65$)

#### **Confusion Matrix**
```text
[[ 53   0]  <- True Negatives | False Positives
 [  1 146]] <- False Negatives| True Positives
```

---

## Feature Importance (Gain)
According to the model, Placement Readiness and SQL are the top predictors, likely due to their high correlation with overall student performance.

---


## Installation & Setup

Clone the repository:

```Bash
git clone [https://github.com/RoBiul-Hasan-Jisan/Student-Placement-Predictor.git](https://github.com/your-username/Student-Placement-Predictor.git)
cd Student-Placement-Predictor
```

---

Install Dependencies:



```Bash
pip install -r requirements.txt

```
---

Run the Flask Web App:


```Bash
python app.py
```

---

Access the app at http://localhost:5000
### Live Link :
https://student-placement-predictor-1-odh2.onrender.com/


## How It Works

- Input: User enters raw scores for technical and soft skills.

- Transformation: The system applies the pre-defined weights ( doubling the Maths score).

- Inference: The LightGBM model processes the weighted vector.

- Thresholding: If the probability exceeds 0.65, the student is flagged as PLACED.
