import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Customer Churn Prediction</h1>
    <p>Using <b>Logistic Regression</b> to predict whether a customer is likely to churn or stay</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# ---------------- DATA PREVIEW ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPROCESSING ----------------
df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ---------------- SELECT IMPORTANT FEATURES ----------------
df_model = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "Churn"]]

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# ---------------- CONFUSION MATRIX ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PERFORMANCE METRICS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("True Positive (TP)", TP)

c3, c4 = st.columns(2)
c3.metric("True Negative (TN)", TN)
c4.metric("False Positive (FP)", FP)

c5, c6 = st.columns(2)
c5.metric("False Negative (FN)", FN)
c6.metric("Total Predictions", len(y_test))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- EXPLANATION BOX ----------------
st.markdown("""
<div class="card">
<h3>Confusion Matrix Meaning</h3>
<ul>
<li><b>TP (True Positive):</b> Correctly identified churn customers</li>
<li><b>TN (True Negative):</b> Correctly identified non-churn customers</li>
<li><b>FP (False Positive):</b> Non-churn predicted as churn</li>
<li><b>FN (False Negative):</b> Churn predicted as non-churn</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Customer Churn")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Manual Encoding (same order as training)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

input_data = np.array([
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    internet_map[internet_service]
]).reshape(1, -1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

result = "Likely to Churn" if prediction == 1 else "Likely to Stay"

st.markdown(f"""
<div class="prediction-box">
Prediction: <b>{result}</b><br>
Churn Probability: <b>{probability:.2f}</b>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)