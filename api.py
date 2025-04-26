from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

model = load_model('plant_disease_model.h5')

DISEASE_CLASSES = ["Leaf Mold", "Blight", "Rust", "Healthy"]
NPK_CLASSES = ["Balanced", "Nitrogen Deficient", "Phosphorus Deficient"]
ROOT_CLASSES = ["Healthy", "Root Rot"]

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        image_data = base64.b64decode(data['image_base64'])
        img = image.load_img(BytesIO(image_data), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        disease_pred, npk_pred, root_pred = model.predict(img_array)

        disease_label = DISEASE_CLASSES[np.argmax(disease_pred)]
        npk_label = NPK_CLASSES[np.argmax(npk_pred)]
        root_label = ROOT_CLASSES[np.argmax(root_pred)]

        # Disease explanation info
        DISEASE_INFO = {
            "Leaf Mold": {
                "cause": "Fungal infection due to humid environments.",
                "treatment": "Improve airflow, reduce humidity, apply fungicide."
            },
            "Blight": {
                "cause": "Caused by various fungi and bacteria under wet conditions.",
                "treatment": "Prune infected leaves, apply copper-based fungicides."
            },
            "Rust": {
                "cause": "Caused by rust fungi, forms orange-brown pustules.",
                "treatment": "Remove infected parts, use sulfur or fungicides."
            },
            "Healthy": {
                "cause": "No disease detected.",
                "treatment": "No treatment needed."
            }
        }

        info = DISEASE_INFO.get(disease_label, {
            "cause": "No info available.",
            "treatment": "Consult a specialist."
        })

        return jsonify({
            "disease": disease_label,
            "cause": info['cause'],
            "treatment": info['treatment'],
            "npkStatus": npk_label,
            "rootCondition": root_label
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
