import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

st.set_option("deprecation.showPyplotGlobalUse", False)

df = pd.read_csv("heart.csv")

# Inisialisasi st.session_state
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "clf_nb" not in st.session_state:
    st.session_state.clf_nb = None


def preprocessing(df):
    # DROP OUTLIER
    for i in df.columns:
        if df[i].nunique() >= 12:
            Q1 = df[i].quantile(0.25)
            Q3 = df[i].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[i] >= (Q1 - 1.5 * IQR)) & (df[i] <= (Q3 + 1.5 * IQR))]

    df = df.reset_index(drop=True)

    # SMOTE
    xf = df.columns
    X = df.drop(["target"], axis=1)
    Y = df["target"]

    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)

    df = pd.DataFrame(X, columns=xf)
    df["target"] = Y

    # MRMR
    np.random.seed(42)
    mi = mutual_info_classif(df.iloc[:, :-1], df.iloc[:, -1])
    redundancy = np.zeros(mi.shape)
    for i in range(mi.shape[0]):
        for j in range(mi.shape[0]):
            if i != j:
                redundancy[i] += mi[i] * mi[j] / mi.sum()

    mrmr = mi - redundancy

    hapus = np.argsort(mrmr)[:3]
    df.drop(df.columns[hapus], axis=1, inplace=True)

    # MINMAX SCALER
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled["target"] = df["target"]
    df = df_scaled

    return df


def modeling():
    if st.session_state.preprocessed_data is not None:
        df_preprocessed = st.session_state.preprocessed_data
        y = df_preprocessed["target"]
        X = df_preprocessed.drop("target", axis=1)

        dtrain_percentage = st.slider(
            "Masukkan data Training (%)", min_value=10, max_value=100, step=1, value=80
        )

        Lanjut = st.button("Lanjut")

        if Lanjut:
            dtrain = dtrain_percentage / 100.0

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, y, train_size=dtrain, random_state=42
            )
            clf_nb = GaussianNB()
            clf_nb.fit(X_train, Y_train)

            Y_pred = clf_nb.predict(X_test)

            clf_nb_acc = accuracy_score(Y_test, Y_pred)
            st.write("Akurasinya adalah:", clf_nb_acc)

            st.session_state.X_test = X_test
            st.session_state.Y_test = Y_test

            st.session_state.clf_nb = clf_nb
    else:
        st.warning("Harap lakukan Preprocessing Data terlebih dahulu")


def evaluation():
    if st.session_state.clf_nb is not None:
        clf_nb = st.session_state.clf_nb

        clf_nb = st.session_state.clf_nb
        X_test = st.session_state.X_test
        Y_test = st.session_state.Y_test

        Y_pred = clf_nb.predict(X_test)

        # Classification Report
        st.caption("Classification Report")
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        f_accuracy = "{:.2f}".format(accuracy)
        f_precision = "{:.2f}".format(precision)
        f_recall = "{:.2f}".format(recall)
        f_f1 = "{:.2f}".format(f1)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f_accuracy)
        col2.metric("Precision", f_precision)
        col3.metric("Recall", f_recall)
        col4.metric("F1 Score", f_f1)
        st.divider()

        st.write("Confusion Matrix:")
        cm = confusion_matrix(Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        st.pyplot()

    else:
        st.warning("Harap selesaikan tahap Modeling terlebih dahulu")


def prediction():
    if st.session_state.clf_nb is not None:
        df_preprocessed = st.session_state.preprocessed_data

        st.subheader("Masukkan Fitur untuk Prediksi:")
        feature_values = {}

        for column in df_preprocessed.columns:
            if column == "target":
                continue

            # st.write(f"### {column}")

            if column == "age":
                feature_values[column] = st.slider(
                    f"{column} (20-90):", min_value=20, max_value=90, value=45
                )
            elif column == "sex":
                sex_options = {"Perempuan": 0, "Laki-laki": 1}
                selected_sex = st.selectbox(f"{column}:", list(sex_options.keys()))
                feature_values[column] = sex_options[selected_sex]
            elif column == "cp":
                cp_options = {"Tidak ada": 0, "Ringan": 1, "Signifikan": 2, "Hebat": 3}
                selected_cp = st.selectbox(f"{column}:", list(cp_options.keys()))
                feature_values[column] = cp_options[selected_cp]
            elif column == "trestbps":
                feature_values[column] = st.slider(
                    f"{column} (90-200):", min_value=900, max_value=200, value=120
                )
            elif column == "chol":
                feature_values[column] = st.slider(
                    f"{column} (100-600):", min_value=100, max_value=600, value=200
                )
            elif column == "fbs":
                fbs_options = {"< 120": 0, "> 120": 1}
                selected_fbs = st.selectbox(f"{column}:", list(fbs_options.keys()))
                feature_values[column] = fbs_options[selected_fbs]
            elif column == "restecg":
                restecg_options = {"Normal": 0, "Abnormal": 1, "Hipertrofi": 2}
                selected_restecg = st.selectbox(
                    f"{column}:", list(restecg_options.keys())
                )
                feature_values[column] = restecg_options[selected_restecg]
            elif column == "thalach":
                feature_values[column] = st.slider(
                    f"{column} (100-200):", min_value=100, max_value=200, value=150
                )
            elif column == "exang":
                exang_options = {"Tidak": 0, "Iya": 1}
                selected_exang = st.selectbox(f"{column}:", list(exang_options.keys()))
                feature_values[column] = exang_options[selected_exang]
            elif column == "oldpeak":
                feature_values[column] = st.slider(
                    f"{column} (0.0-4.0):", min_value=0.0, max_value=4.0, value=1.0
                )
            elif column == "slope":
                slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
                selected_slope = st.selectbox(f"{column}:", list(slope_options.keys()))
                feature_values[column] = slope_options[selected_slope]
            elif column == "ca":
                ca_options = {
                    "Tidak ada": 0,
                    "Normal": 4,
                    "1 arteri": 1,
                    "2 arteri": 2,
                    "3 arteri": 3,
                }
                selected_ca = st.selectbox(f"{column}:", list(ca_options.keys()))
                feature_values[column] = ca_options[selected_ca]
            elif column == "thal":
                thal_options = {
                    "Tidak ada": 0,
                    "Normal": 1,
                    "Fixed Defect": 2,
                    "Rever Defect": 3,
                }
                selected_thal = st.selectbox(f"{column}:", list(thal_options.keys()))
                feature_values[column] = thal_options[selected_thal]
            else:
                feature_values[column] = st.number_input(f"{column}:")

        if st.button("Prediksi"):
            input_data = pd.DataFrame([feature_values])

            input_data = input_data[df_preprocessed.drop(columns=["target"]).columns]
            scaler = MinMaxScaler()
            scaler.fit(df_preprocessed.drop(columns=["target"]))
            input_data_scaled = scaler.transform(input_data)
            input_data_scaled = pd.DataFrame(
                input_data_scaled, columns=input_data.columns
            )

            clf_nb = st.session_state.clf_nb
            prediction = clf_nb.predict(input_data_scaled)[0]

            st.write("Hasil Prediksi:", prediction)

    else:
        st.warning("Harap selesaikan tahap Modeling terlebih dahulu")


def show_preprocessing():
    st.session_state.preprocessed_data = preprocessing(df)
    st.write(st.session_state.preprocessed_data)


# Sidebar
with st.sidebar:
    selected = option_menu(
        "Prediksi Penyakit Jantung",
        ["Home", "Dataset", "Preprocessing", "Modeling", "Evaluation", "Prediction"],
        icons=[
            "house",
            "clipboard-data",
            "gear",
            "clipboard2-heart",
            "activity",
            "search-heart",
        ],
        default_index=0,
    )

if selected == "Home":
    st.title(
        "IMPLEMENTASI MACHINE LEARNING UNTUK MEMPREDIKSI PENYAKIT JANTUNG MENGGUNAKAN ALGORITMA NAÏVE BAYES"
    )

elif selected == "Dataset":
    st.title("Raw Dataset")
    st.write(df)

elif selected == "Preprocessing":
    st.title("Preprocessing Data")
    show_preprocessing()

elif selected == "Modeling":
    st.title("Model Building")
    modeling()

elif selected == "Evaluation":
    st.title("Model Evaluation")
    evaluation()

elif selected == "Prediction":
    st.title("Prediction")
    prediction()
