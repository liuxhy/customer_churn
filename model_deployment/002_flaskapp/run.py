from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd
from utils import Transformer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        gender = request.form["gender"]
        SeniorCitizen = request.form["SeniorCitizen"]
        Partner = request.form["Partner"]
        Dependents = request.form["Dependents"]
        tenure = request.form["tenure"]
        InternetService = request.form["InternetService"]
        PhoneService = request.form["PhoneService"]
        MultipleLines = request.form["MultipleLines"]
        OnlineSecurity = request.form["OnlineSecurity"]
        OnlineBackup = request.form["OnlineBackup"]
        DeviceProtection = request.form["DeviceProtection"]
        TechSupport = request.form["TechSupport"]
        StreamingTV = request.form["StreamingTV"]
        StreamingMovies = request.form["StreamingMovies"]
        PaperlessBilling = request.form["PaperlessBilling"]
        Contract = request.form["Contract"]
        PaymentMethod = request.form["PaymentMethod"]
        MonthlyCharges = request.form["MonthlyCharges"]
        TotalCharges = request.form["TotalCharges"]

        X = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'InternetService': [InternetService],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'PaperlessBilling': [PaperlessBilling],
            'Contract': [Contract],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        })
        
        pred = model.predict_proba(X)[0][1]
        
    return render_template("index.html", pred=pred)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
