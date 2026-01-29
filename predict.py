import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import pandas as pd

# Load model and class names
model = tf.keras.models.load_model('models/nepal_monument_model.keras')
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Load and preprocess image
img_path = 'test_images/patan.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)[0]
top_idx = np.argmax(predictions)

predicted_monument_id = class_names[top_idx]
print(f"Predicted Monument ID: {predicted_monument_id}")
print(f"Confidence: {predictions[top_idx]*100:.2f}%")


#To get monument details from monument_details.csv file
monument_df = pd.read_csv("data/monument_details.csv", encoding='latin1')

#returns monument information
def get_monument_info(monument_id, monument_df):
    monument_id = int(monument_id)
    row = monument_df[monument_df["monument_id"] == monument_id]
    type(row)
    return row.iloc[0] if not row.empty else None

# Monument info
info = get_monument_info(predicted_monument_id, monument_df)

if info is not None:
    print("\nMonument Information")
    print(f"Name       : {info['monument_name']}")
    print(f"Location   : {info['location']}")
    print(f"Built      : {info['built']}")
    print(f"Built By   : {info['built_by']}")
    print(f"Description: {info['description']}")
    print(f"Source     : {info['source']}")
else:
    print("\nNo historical information found.")