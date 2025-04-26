import base64
import requests
from time import sleep
from picamera import PiCamera

#Config 
API_URL = "http://<YOUR-PC-IP>:5000/api/predict"
IMAGE_PATH = "leaf.jpg"

camera = PiCamera()
camera.resolution = (640, 480)
camera.start_preview()
sleep(2) 
camera.capture(IMAGE_PATH)
camera.stop_preview()

with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post(API_URL, json={"image_base64": encoded_image})

if response.status_code == 200:
    result = response.json()
    print("\n🌿 AI Prediction Result:")
    print(f"  Disease:        {result['disease']}")
    print(f"  Cause:          {result['cause']}")
    print(f"  Treatment:      {result['treatment']}")
    print(f"  NPK Status:     {result['npkStatus']}")
    print(f"  Root Condition: {result['rootCondition']}")
else:
    print("❌ Error:", response.json().get("error", "Unknown error"))
