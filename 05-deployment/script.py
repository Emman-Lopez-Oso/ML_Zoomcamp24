#! Question 3 script

import pickle

with open("model1.bin", "rb") as model_file:
    model = pickle.load(model_file)

with open("dv.bin", "rb") as dv_file:
    dv = pickle.load(dv_file)

# Define the client data
client = {
    # Example data - replace with actual client data as required
   "job": "management", 
   "duration": 400, 
   "poutcome": "success"}

# Transform client data and make prediction
X = dv.transform([client])  # Transform using the DictVectorizer
prediction = model.predict_proba(X)[0, 1]  # Get probability for class 1

print(f"Probability of class 1 for this client: {round(prediction,3)}")