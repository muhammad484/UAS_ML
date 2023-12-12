import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset (asumsi dataset dalam format CSV)
@st.cache
def load_data():
    df = pd.read_csv("heart_failure_dataset.csv")  # Ganti dengan nama file dataset yang sesuai
    return df

# Sidebar
st.sidebar.title("K-NN Heart Failure Survival Prediction")
st.sidebar.markdown("Select the parameters:")

# Load dataset
df = load_data()

# Features and target variable
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
k_value = st.sidebar.slider("Select the number of neighbors (k)", 1, 20, 5)
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.text(f"Model Accuracy: {accuracy:.2%}")

# User input for prediction
st.title("Heart Failure Survival Prediction")
st.markdown("Please enter the following information for prediction:")

age = st.slider("Age", min_value=18, max_value=100, value=50)
creatinine_phosphokinase = st.slider("Creatinine Phosphokinase", min_value=10, max_value=1000, value=250)
ejection_fraction = st.slider("Ejection Fraction", min_value=10, max_value=80, value=35)
platelets = st.slider("Platelets", min_value=50000, max_value=300000, value=150000)
serum_creatinine = st.slider("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.0)
serum_sodium = st.slider("Serum Sodium", min_value=100, max_value=150, value=135)
time = st.slider("Follow-up Period (Time)", min_value=1, max_value=300, value=150)

# Make prediction
input_data = np.array([age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium, time]).reshape(1, -1)
prediction = model.predict(input_data)

# Display prediction
st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.success("The patient is predicted to survive.")
else:
    st.error("The patient is predicted not to survive.")

# Display the dataset
st.subheader("Heart Failure Dataset")
st.write(df)

# Note: Pastikan Anda mengganti nama file dataset dan menyesuaikan feature sesuai dengan dataset yang Anda gunakan.
