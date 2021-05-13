import json
import requests

url = 'https://127.0.0.1:8000/model'

request_data = json.dumps({'age': 40, 'salary': 50000})
response = requests.post(url, request_data)
print(response.text)
