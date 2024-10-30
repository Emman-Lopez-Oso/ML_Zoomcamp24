# Pull all of the libraries in
from flask import Flask, request, jsonify
import pickle

# Read in models with pickle
with open("model1.bin", "rb") as model_file:
    model = pickle.load(model_file)

with open("dv.bin", "rb") as dv_file:
    dv = pickle.load(dv_file)

# Create flask artifact 
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    # Transform client data and make prediction
    X = dv.transform([client])  
    prediction = model.predict_proba(X)[0, 1] 
    return jsonify({"Probability": round(prediction,3)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
