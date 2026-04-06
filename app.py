from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return "Price Prediction API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get input features
        input_data = data["features"]

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# Run app
if __name__ == "__main__":
    app.run(debug=True)