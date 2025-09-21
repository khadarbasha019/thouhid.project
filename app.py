from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model & encoders
MODEL_PATH = os.path.join("data", "delay_model.pkl")
ENCODERS_PATH = os.path.join("data", "label_encoders.pkl")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect input
        input_data = {
            "Country": request.form["country"],
            "Destination_Port": request.form["port"],
            "Commodity": request.form["commodity"],
            "Shipment_Value": float(request.form["value"]),
            "Mode": request.form["mode"],
            "Docs_Submitted": request.form["docs"],
            "Clearance_Time": int(request.form["clearance"]),
            "Delay_Reason": "None",          # Placeholder (not available at input)
            "Delay_Duration": 0,             # Placeholder
            "Alternative_Route": "Direct",   # Placeholder
            "Estimated_Cost": float(request.form["value"]) # approx same as value
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Apply label encoding to categorical columns
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        # Match training features
        X_train_cols = pd.read_csv("data/X_train.csv").columns
        df = df.reindex(columns=X_train_cols, fill_value=0)

        # Predict
        prediction = model.predict(df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
