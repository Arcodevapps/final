from io import BytesIO
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# Load the ONNX model
ort_session = ort.InferenceSession('vit_model.onnx')

# Define the image transformations
def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    # Resize the image
    image = image.resize(target_size)
    
    # Convert image to float32 and normalize
    image = np.array(image, dtype=np.float32)
    image /= 255.0
    
    # Apply normalization mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image -= mean
    image /= std
    
    # Convert image to CHW format
    image = image.transpose(2, 0, 1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def pred_and_plot_image(ort_session: ort.InferenceSession, class_names: List[str], image_path: str) -> str:
    response = requests.get(image_path)
    response.raise_for_status()

    try:
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    preprocessed_image = preprocess_image(image)
    ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions = ort_outs[0]
    pred_label = np.argmax(predictions, axis=1)[0]
    pred_prob = np.max(softmax(predictions, axis=1))

    result = f"Pred: {class_names[pred_label]} | Prob: {pred_prob:.3f}"
    print(result)
    return result

def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)

# Define your class names
class_names = ['Areacanut_healthy',
'Areacanut_inflorecence',
'Areacanut_koleroga',
'Areacnut_natural_aging',
'Arecanut_budroot',
'Arecanut_leafspot',
'Arecanut_suity_mold',
'Arecanut_yellow_leaf',
'Coconut_CCI_Caterpillars',
'Coconut_WCLWD_DryingofLeaflets',
'Coconut_WCLWD_Flaccidity',
'Coconut_WCLWD_Yellowing',
'Coconut_budroot',
'Coconut_healthy_coconut',
'Coconut_rb',
'Coconut_whitefly']


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_path = data["image_path"]
    output = pred_and_plot_image(ort_session, class_names, image_path)  # Pass image_path instead of class_names
    return jsonify({"result": output})

@app.route('/')
def hello():
    return "<h1>Model is ready for prediction</h1>"

if __name__ == '__main__':
    app.run(debug=True)
