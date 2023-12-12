import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (contoh menggunakan dataset Gagal Jantung dari kaggle)
url = "https://raw.githubusercontent.com/andrychowanda/heart-failure-prediction/main/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(url)

# Preprocessing dataset
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("Aplikasi Prediksi Kelangsungan Hidup Pasien Gagal Jantung")
    
    # Input form
    age = st.slider("Usia Pasien", min_value=20, max_value=100, value=50)
    creatinine_phosphokinase = st.slider("Kreatinin Phosphokinase", min_value=10, max_value=1000, value=200)
    ejection_fraction = st.slider("Ejection Fraction", min_value=10, max_value=80, value=40)
    platelets = st.slider("Jumlah Platelets", min_value=50000, max_value=300000, value=150000)
    serum_creatinine = st.slider("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.0)
    serum_sodium = st.slider("Serum Sodium", min_value=100, max_value=150, value=135)
    
    # Predict button
    if st.button("Prediksi Kelangsungan Hidup"):
        input_data = [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]]
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 0:
            st.success(f"Pasien diperkirakan akan hidup dengan probabilitas {probability[0]:.2f}")
        else:
            st.error(f"Pasien diperkirakan tidak akan hidup dengan probabilitas {probability[0]:.2f}")

if __name__ == "__main__":
    main()
