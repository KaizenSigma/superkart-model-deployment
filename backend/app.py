import io
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load("superkart_model.joblib")

required_columns = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@superkart_api.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@superkart_api.post("/v1/predict")
def predict_online():
    payload = request.get_json(force=True)
    input_df = pd.DataFrame([payload])

    missing = [col for col in required_columns if col not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    input_df = input_df[required_columns]
    prediction = model.predict(input_df)[0]

    return jsonify({"predicted_sales": float(prediction)})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    uploaded_file = request.files["file"]
    batch_df = pd.read_csv(io.BytesIO(uploaded_file.read()))

    missing = [col for col in required_columns if col not in batch_df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    batch_df = batch_df[required_columns]
    preds = model.predict(batch_df)

    prediction_map = {int(idx): float(pred) for idx, pred in enumerate(preds)}
    return jsonify(prediction_map)


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
