# Nepal Historical Monument Recognition Using Deep Learning
A deep learning-based system for automated identification of historical monuments in Nepal 
using transfer learning with MobileNetV2 architecture. This project leverages computer vision and convolutional neural networks to recognize 
Nepalese monuments from images, achieving 86.93% validation accuracy with efficient mobile deployment capabilities.

## Monument Classes Include:

1. Muktinath (Id: 98003)
2. Patan Durbar Square (Id: 4137)
3. Janaki Temple (Id: 7922)
4. Pashupatinath  (Id: 152527)
5. Boudhanath Stupa (Id: 168267)
6. Nyatapola Temple (Id: 122791)

## Quick Start Guide - Nepal Historical Monument Recognition
This guide will help to get started with the Nepal Historical Monument Recognition model

**Step 1: Download Required Files**
* nepal_monument_model.keras (13 MB Trained Model)
* class_names.json (class mapping file)
* nepal_monuments.csv (Monument details)
* test_model.ipynb ( (optional) code for testing for the guidance)
* test_images ( (optional) folder that contains test images)

**Step 2: Sample Folder Structure**
```
  project/
  ├── models/
  │   ├── nepal_monument_final.keras
  │   └── class_names.json
  ├── data/
  │   └── nepal_monuments.csv
  ├── predict.py
  └── test_images/
      └── image.jpg
```
*predict.py is the python file that tests the model*

**Step 3: Install Dependencies**

```python 
pip install -r requirements.txt
```

**Step 4: Prediction**
<br>Make predict.py file and use the following code: 
```python 
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
```

**Note before prediction making:** <br>
1. Need to resize the image to 224 x 224
2. Need to normalize the image by dividing it with 255

**To Run**
```
python predict.py
```

**Expected Result**
```
Predicted Monument ID: 4137
Confidence: 99.90%

Monument Information
Name       : Patan Dubar Square
Location   : Lalitpur
Built      : 1637
Built By   : Malla dynasty
Description: Patan Durbar Square (Nepal Bhasa: ???? ???????????/?? ?????, Nepali: ???? ????? ???????) is situated at the centre of the city 
of Lalitpur in Nepal. It is one of the three Durbar Squares in the Kathmandu Valley, all of which are UNESCO World Heritage Sites. One of its attractions is the medieval royal palace where the Malla Kings of Lalitpur resided.
Source     : http://www.patanmuseum.gov.np/
```
