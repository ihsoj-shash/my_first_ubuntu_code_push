import json
import requests

url = 'https://127.0.0.2:8000/model'

request_data = json.dumps({'model': 'knn'})
response = requests.post(url, request_data)
print(response.text)
