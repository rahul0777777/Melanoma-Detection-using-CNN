import tensorflow as tf
from tensorflow.keras import layers,models
import os
import pandas as pd
import cv2
import numpy as np
import pickle

new2=cv2.imread('C:/Users/ASUS/data/test/malignant/1499.jpg')
df=pd.DataFrame(columns=['images'])
df.loc[len(df.index)] = [new2]
x2=df.iloc[:,0]
x2=np.stack(x2).astype(None)
x2=x2/255.0

with open('cnn.pkl', 'rb') as file:
    # Load the data from the pickle file
    cnn = pickle.load(file)
y_pred=cnn.predict(x2)
out_class=[np.argmax(element) for element in y_pred]
print(out_class[0])