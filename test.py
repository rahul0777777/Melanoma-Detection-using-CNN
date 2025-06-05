import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import cv2
import numpy as np
from preprocessing1 import grayScaleConversion, dullRazor, noiseRemoval, imageEnhancement, segmentation

def preprocess_images_from_directory(directory):
    # Perform preprocessing steps
    dict1, dict2 = grayScaleConversion(directory)
    dict1, dict2 = dullRazor(dict1,dict2)
    dict1, dict2 = noiseRemoval(dict1, dict2)
    dict1, dict2 = imageEnhancement(dict1, dict2)
    dict1, dict2 = segmentation(dict1, dict2)

    # Combine dictionaries and convert to DataFrame
    image_dict = {**dict1, **dict2}
    df = pd.DataFrame(columns=['images'])
    for key, value in image_dict.items():
        img = cv2.resize(value, (224, 224))
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
        df.loc[len(df.index)] = [img]

    return df

# Load and preprocess images from the test directory
test_directory = 'C:/new project file/inputs/test'
df_test = preprocess_images_from_directory(test_directory)

x_test = np.array(df_test['images'].tolist()).astype('float32')
x_test = x_test / 255.0

# Debugging: Print shape and type of x_test
print("Shape of x_test:", x_test.shape)
print("Data type of x_test:", x_test.dtype)

# Load the trained model
try:
    cnn = keras.models.load_model('cnn.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Debugging: Print model summary to verify architecture
try:
    print(cnn.summary())
except Exception as e:
    print(f"Error printing model summary: {e}")

# Predict classes for the test images
try:
    y_pred = cnn.predict(x_test)
    out_classes = [np.argmax(element) for element in y_pred]

    # Print predictions for all images including those in subdirectories
    idx = 0
    for root, dirs, files in os.walk(test_directory):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print(f"Image: {filename}, Predicted class: {out_classes[idx]}")
                idx += 1
except Exception as e:
    print(f"Error during prediction: {e}")
