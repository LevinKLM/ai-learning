import requests
import os
import json

base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

try:
    print(f"Checking models at {base_url}...")
    response = requests.get(f"{base_url}/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get('models', [])
        print("Available models:")
        for m in models:
            print(f"- {m['name']}")
    else:
        print(f"Error listing models: {response.status_code} {response.text}")
except Exception as e:
    print(f"Failed to connect: {e}")
