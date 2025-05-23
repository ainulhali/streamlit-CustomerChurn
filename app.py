import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input form
st.title("Prediksi Churn Pelanggan")
st.write("Masukkan fitur-fitur pelanggan:")

col1, col2 = st.columns(2)

with col1:
    feature_0 = st.number_input("Fitur 1", min_value=0.0, max_value=1.0)
    feature_1 = st.number_input("Fitur 2", min_value=0.0, max_value=1.0)

with col2:
    feature_2 = st.number_input("Fitur 3", min_value=0.0, max_value=1.0)
    feature_3 = st.number_input("Fitur 4", min_value=0.0, max_value=1.0)

input_data = np.array([[feature_0, feature_1, feature_2, feature_3]])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    st.success(f"Hasil Prediksi: {'Churn' if prediction == 1 else 'Tidak Churn'}")

    # Simulasi confusion matrix
    X_dummy = np.random.rand(30, 4)
    y_true = np.random.randint(0, 2, 30)
    y_pred = model.predict(X_dummy)

    cm = confusion_matrix(y_true, y_pred)
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
