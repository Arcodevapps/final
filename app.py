import json
import threading
from io import BytesIO
from typing import List, Tuple

import firebase_admin
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from firebase_admin import credentials, db, firestore
from PIL import Image

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
        error = {"Pred": "Imporper Image" , "Prob": "Please upload correct image","Susp":"Wrong image"}
        return error

    preprocessed_image = preprocess_image(image)
    ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions = ort_outs[0]
    pred_label = np.argmax(predictions, axis=1)[0]
    pred_prob = np.max(softmax(predictions, axis=1))

    result = {"Pred": class_names[pred_label] , "Prob": round(pred_prob * 100, 2),"Susp":class_names[pred_label]}
    notfound_dict = {
            "Pred": "Improper Image", "Susp":class_names[pred_label],
            "Prob": round(pred_prob * 100, 2)
        }

    if (pred_prob<=0.65):
        return json.dumps(notfound_dict)
    else:
        return json.dumps(result)

    
            ##"Prob": round(max_prob.item() * 100, 2)

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


def on_snapshot(col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED' :  # Check for added or modified documents
            doc_ref = change.document.reference
            doc = doc_ref.get().to_dict()
            print("Added: "+change.document.id)
            
        if  change.type.name == 'MODIFIED':
            doc_ref = change.document.reference
            doc = doc_ref.get().to_dict()
            print("Data Modified DocId: "+change.document.id)
            colname = doc.get('username')  # Retrieve the collection name
            if colname:
                # Retrieve the document from the corresponding collection and update it
                col_ref = db.collection(colname)
                
                docs = col_ref.get()
                # Update each document with the new field "Pred"
                for doc in docs:
                    doc_data = doc.to_dict()
                    
                    if 'img' in doc_data.keys() and 'Prediction' not in doc_data.keys():
                        
                        image_path=doc_data['img']
                        
                        output="sample"
                        print("New image Added")
                        # Predict on custom image
                        output = pred_and_plot_image(ort_session, class_names, image_path)  # Pass image_path instead of class_names

                        y=json.loads(output)
                        
                        data={"Prediction": y["Pred"], "Probability": y["Prob"],"comment":"","Suspected":y["Susp"]}
                        #Update a new doc in collection 
                        doc.reference.update(data)
                        print(data)
                        
    callback_done.set()
    print("Everthing is Updated")

cred = credentials.Certificate('./auth-react-appKey.json')
try:
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    st.write("Firebase OK")

    # Create an Event for notifying main thread.
    callback_done = threading.Event()
    print(callback_done)
    col_query = db.collection("users")
    # Watch the collection query
    query_watch = col_query.on_snapshot(on_snapshot)
except:
    print("Streamlit running")
##############################################################


def main():
    st.title("Server Running Successfully")

    
    # Button to generate a random number
    
    st.title(f"All OK Team ARCO")
    st.write(f"ARCO DevApps")

if __name__ == "__main__":
    main()
