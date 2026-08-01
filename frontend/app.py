import io
import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("SuperKart Sales Forecasting App")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.subheader("Single Prediction")

col1, col2 = st.columns(2)

with col1:
    product_weight = st.number_input("Product_Weight", min_value=0.0, value=12.66)
    product_sugar = st.selectbox("Product_Sugar_Content", ["Low Sugar", "Regular", "No Sugar"])
    product_alloc_area = st.number_input("Product_Allocated_Area", min_value=0.0, max_value=1.0, value=0.027)
    product_mrp = st.number_input("Product_MRP", min_value=1.0, value=117.08)
    store_size = st.selectbox("Store_Size", ["Small", "Medium", "High"])

with col2:
    city_tier = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store_Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Food Mart"])
    product_id_char = st.selectbox("Product_Id_char", ["FD", "DR", "NC"])
    store_age = st.number_input("Store_Age_Years", min_value=0, value=16)
    product_type_category = st.selectbox("Product_Type_Category", ["Perishables", "Non Perishables"])

if st.button("Predict Sales"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_alloc_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_tier,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_type_category,
    }

    response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
    if response.status_code == 200:
        st.success(f"Predicted Sales: {response.json()['predicted_sales']:.2f}")
    else:
        st.error(f"Error: {response.text}")

st.divider()
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload batch CSV", type=["csv"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)

    if response.status_code == 200:
        pred_dict = response.json()
        pred_df = pd.DataFrame(
            {
                "row_index": list(pred_dict.keys()),
                "predicted_sales": list(pred_dict.values()),
            }
        )
        st.dataframe(pred_df)
    else:
        st.error(f"Error: {response.text}")
