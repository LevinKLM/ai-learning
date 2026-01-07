import requests
import os
import json

base_url = "http://172.27.128.1:11434"

try:
    # Check what models are currently loaded/running
    response = requests.get(f"{base_url}/api/ps", timeout=5)
    if response.status_code == 200:
        running = response.json().get('models', [])
        print(f"Running Models ({len(running)}):")
        for m in running:
            print(f"- {m['name']} (Size: {m['size_vram']/1024/1024/1024:.2f} GB)")
    else:
        print(f"Error checking status: {response.status_code}")

except Exception as e:
    print(f"Failed to connect: {e}")
