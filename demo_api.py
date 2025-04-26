# # from flask import Flask, request, jsonify
# # from flask_cors import CORS

# # app = Flask(__name__)
# # CORS(app)

# # @app.route('/api/predict', methods=['POST'])
# # def predict():
# #     file = request.files.get('file')
# #     if not file:
# #         return jsonify({'error': 'No file uploaded'}), 400

# #     # 🔮 Replace this block with your ML model later
# #     mock_result = {
# #         "disease": "Leaf Spot",
# #         "cause": "Fungal infection due to humid conditions.",
# #         "treatment": "Use a fungicide and ensure good air circulation.",
# #         "waterNeeded": False
# #     }

# #     return jsonify(mock_result)

# # if __name__ == '__main__':
# #     app.run(debug=True)


# import tensorflow as tf
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from io import BytesIO
# #from tensorflow.keras.preprocessing.image import load_img
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# # Load model once at startup
# model = load_model('plant_disease_model.h5')

# # Your class labels must match model output order
# CLASS_NAMES = [
#     "Apple Scab", "Apple Black Rot", "Corn Rust", "Corn Gray Leaf Spot",
#     "Healthy", "Potato Early Blight", "Potato Late Blight", "Tomato Leaf Mold"
# ]

# DISEASE_INFO = {
#     "Apple Scab": {
#         "cause": "Caused by the fungus *Venturia inaequalis*, often due to wet spring conditions.",
#         "treatment": "Apply fungicides, prune infected areas, and avoid overhead watering."
#     },
#     "Apple Black Rot": {
#         "cause": "Caused by the fungus *Botryosphaeria obtusa*, common in warm, humid areas.",
#         "treatment": "Remove and destroy infected fruit and branches, use copper-based sprays."
#     },
#     "Corn Rust": {
#         "cause": "Fungal infection due to *Puccinia sorghi* spores.",
#         "treatment": "Use resistant corn varieties and apply appropriate fungicides."
#     },
#     "Corn Gray Leaf Spot": {
#         "cause": "Caused by *Cercospora zeae-maydis*, thrives in hot and humid conditions.",
#         "treatment": "Rotate crops, manage irrigation, and use resistant hybrids."
#     },
#     "Healthy": {
#         "cause": "No disease detected.",
#         "treatment": "No treatment necessary. Continue regular care and monitoring."
#     },
#     "Potato Early Blight": {
#         "cause": "Caused by *Alternaria solani*, often after periods of wet weather.",
#         "treatment": "Use fungicides and remove infected leaves promptly."
#     },
#     "Potato Late Blight": {
#         "cause": "Caused by *Phytophthora infestans*, spreads quickly in cool, moist environments.",
#         "treatment": "Remove affected plants and use systemic fungicides."
#     },
#     "Tomato Leaf Mold": {
#         "cause": "Fungal disease caused by *Passalora fulva*, thrives in warm, moist greenhouses.",
#         "treatment": "Improve ventilation, remove infected leaves, and apply sulfur sprays."
#     }
# }

# @app.route('/api/predict', methods=['POST'])

# def predict():
#     file = request.files['file']
#     if not file:
#         return jsonify({'error': 'No file uploaded'}), 400

#     # Preprocess the image
#     # img = image.load_img(file, target_size=(224, 224))  # or your model's input size
    
#     img = image.load_img(BytesIO(file.read()), target_size=(256, 256))  # RGB by default
#     img_array = image.img_to_array(img)  # Shape: (256, 256, 3)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0  # Shape: (1, 256, 256, 3)


#     prediction = model.predict(img_array)[0]
#     class_index = np.argmax(prediction)
    
#     confidence = np.max(prediction)

#     if confidence < 1e-5:
#         return jsonify({
#             "disease": "Unknown",
#             "cause": "The image doesn't appear to be a plant leaf or is unclear.",
#             "treatment": "Try uploading a clear photo of a single plant leaf.",
#             "waterNeeded": None
#         })
    
#     predicted_label = CLASS_NAMES[class_index]
    
#     # # Fake treatment data – replace with real advice
#     # info = {
#     #     "Apple Scab": {
#     #         "cause": "Fungal infection",
#     #         "treatment": "Use fungicides and remove infected leaves."
#     #     },
#     #     "Healthy": {
#     #         "cause": "None",
#     #         "treatment": "Plant is healthy!"
#     #     },
#     #     # ... add more
#     # }

#     # response = {
#     #     "disease": predicted_label,
#     #     "cause": info.get(predicted_label, {}).get("cause", "Unknown"),
#     #     "treatment": info.get(predicted_label, {}).get("treatment", "Consult a specialist."),
#     #     "waterNeeded": predicted_label != "Healthy"  # Example logic
#     # }
    
#     info = DISEASE_INFO.get(predicted_label, {
#     "cause": "No information available.",
#     "treatment": "Please consult an expert."
# })
    
#     response = {
#         "disease": predicted_label,
#         "cause": info["cause"],
#         "treatment": info["treatment"],
#         "waterNeeded": predicted_label != "Healthy"
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = load_model('plant_disease_model.h5')

DISEASE_CLASSES = ["Leaf Mold", "Blight", "Rust", "Healthy"]
NPK_CLASSES = ["Balanced", "Nitrogen Deficient", "Phosphorus Deficient"]
ROOT_CLASSES = ["Healthy", "Root Rot"]

@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    img = image.load_img(BytesIO(file.read()), target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    disease_pred, npk_pred, root_pred = model.predict(img_array)

    disease_label = DISEASE_CLASSES[np.argmax(disease_pred)]
    npk_label = NPK_CLASSES[np.argmax(npk_pred)]
    root_label = ROOT_CLASSES[np.argmax(root_pred)]

    # Example info dictionary
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

if __name__ == '__main__':
    app.run(debug=True)
