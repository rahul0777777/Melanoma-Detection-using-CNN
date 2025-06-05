import tensorflow as tf
from tensorflow.keras import layers,models
import os
import pandas as pd
import cv2
import numpy as np
import pickle

def makedf(img1, img2, path1, path2):
    df = pd.DataFrame(columns=['images', 'outcome'])

    for image in img1:
        p = path1 + '/' + image
        img = cv2.imread(p)
        df.loc[len(df.index)] = [img, 'benign']

    for image in img2:
        p = path2 + '/' + image
        img = cv2.imread(p)
        df.loc[len(df.index)] = [img, 'malignant']

    return df
path1='C:/Users/ASUS/data/train/benign'
path2='C:/Users/ASUS/data/train/malignant'
images1=os.listdir(path1)
# print(images1)
images2=os.listdir(path2)
df1=makedf(images1,images2,path1,path2)
x=df1.iloc[:,0]
y = df1['outcome'].map({"benign":0,"malignant":1})
y=np.asarray(y)
x=np.stack(x).astype(None)
x=x/255.0
path3='C:/Users/ASUS/data/test/benign'
path4='C:/Users/ASUS/data/test/malignant'
images3=os.listdir(path3)
images4=os.listdir(path4)
df2=makedf(images3,images4,path3,path4)
x_test=df2.iloc[:,0]
y_test = df2['outcome'].map({"benign":0,"malignant":1})
x_test=np.stack(x_test).astype(None)
x_test=x_test/255.0
y_test=np.asarray(y_test)
cnn = models.Sequential([
    # convolutional layer
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnn.fit(x, y, epochs=10)
pickle.dump(cnn, open('cnn.pkl', 'wb'))
y_pred=cnn.predict(x);
print(y_pred)