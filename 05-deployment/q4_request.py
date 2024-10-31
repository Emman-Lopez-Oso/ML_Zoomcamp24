import requests

url = "http://127.0.0.1:9696/predict"  # Or use the Codespace URL
client = {"job": "student", "duration": 280, "poutcome": "failure"}
response = requests.post(url, json=client).json()

print("Probability of class 1 for this client:", response["probability"])