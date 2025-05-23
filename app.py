import streamlit as st
import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset_customerchurn.csv")

# Preprocessing seperti saat training
df = df.dropna()
customer_ids = df['customerID'].tolist()

# Encode target
le_churn = LabelEncoder()
df['Churn'] = le_churn.fit_transform(df['Churn'])

# Pisahkan fitur dan target
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']

# Encode semua kolom kategorikal
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train ulang model (atau load model jika Anda simpan sebagai pickle)
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Prediksi Churn Pelanggan")

selected_id = st.selectbox("Pilih customerID:", customer_ids)

if st.button("Prediksi Churn"):
    # Ambil data berdasarkan ID
    input_data = df[df['customerID'] == selected_id]
    input_X = input_data.drop(columns=['Churn', 'customerID'])

    # Encode sama seperti training
    for col in input_X.select_dtypes(include='object').columns:
        input_X[col] = encoders[col].transform(input_X[col])

    pred = model.predict(input_X)[0]
    result = le_churn.inverse_transform([pred])[0]

    st.success(f"Hasil prediksi: {result}")
