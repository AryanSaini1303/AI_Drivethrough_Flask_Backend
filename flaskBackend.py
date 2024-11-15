from flask import Flask, request, jsonify
import joblib  # Replace with your model loading library
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Load the trained ML model
model = joblib.load("model.pkl")  # Adjust path based on your model file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    features = np.array(data['features']).reshape(1, -1)  # Prepare for model
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})  # Return as JSON

if __name__ == '__main__':
    app.run(debug=True)
