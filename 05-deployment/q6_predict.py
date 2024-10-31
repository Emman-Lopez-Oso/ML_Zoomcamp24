import requests

url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client)

print("Full response:", response.content)

if response.status_code == 200:
    data = response.json()  # Decode JSON content to a dictionary
    print("Probability of subscription:", data.get("Probability", "Key 'probability' not found"))
else:
    print("Error:", response.status_code, response.content)