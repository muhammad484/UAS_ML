# Install library yang diperlukan jika belum terinstall
# !pip install streamlit pandas scikit-learn

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(url)

# Sidebar
st.sidebar.title("Parameter Model")
test_size = st.sidebar.slider("Porsi Data Uji", 0.1, 0.5, 0.2, 0.05)

# Model
def knn_model(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# Main
st.title("Prediksi Kelangsungan Hidup Pasien Gagal Jantung")
st.write("Dataset: [Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv)")

# Tampilkan beberapa baris pertama dataset
st.subheader("Data Preview")
st.write(df.head())

# Pisahkan fitur dan label
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Sidebar untuk parameter k
k = st.sidebar.slider("Jumlah Tetangga (k)", 1, 20, 5, 1)

# Training model
knn_classifier = knn_model(X_train, y_train, k)

# Prediksi
y_pred = knn_classifier.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Tampilkan hasil evaluasi
st.subheader("Evaluasi Model")
st.write(f"Akurasi: {accuracy:.2f}")
st.write("Laporan Klasifikasi:")
st.code(classification_rep)

# Prediksi satu sampel
st.subheader("Prediksi Individu")
sample_data = st.text_input("Masukkan data pasien (cth: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):")
sample_data = [float(i) for i in sample_data.split(',')]
prediction = knn_classifier.predict([sample_data])

if st.button("Prediksi"):
    st.write(f"Prediksi Kelangsungan Hidup: {'Hidup' if prediction[0] == 0 else 'Meninggal'}")
