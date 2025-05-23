import streamlit as st
import json

# Judul aplikasi
st.title("Visualisasi Model Decision Tree")

# Muat file JSON
with open("decision_tree_model.json", "r") as f:
    model_data = json.load(f)

# Tampilkan aturan pohon keputusan
st.subheader("Struktur Model Decision Tree:")
st.text(model_data["model_rules"])
