from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    scaled_data = scaler.transform(np.array(data).reshape(1, -1))
    prediction = model.predict(scaled_data)
    return jsonify({'loan_status_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)