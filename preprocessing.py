import os
import cv2
import numpy as np
from hairRemoval import hairRemoval
def loadRGBImages(path):
    dict1 = {}
    dict2 = {}
    classes = os.listdir(path)
    for classe in classes:
        pathFile = os.path.join(path, classe)
        images = os.listdir(pathFile) 
        for i in range(len(images)):
            img = cv2.imread(os.path.join(pathFile, images[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if classe == 'benign':
                dict1[os.path.join(pathFile, images[i])] = img
            else:
                dict2[os.path.join(pathFile, images[i])] = img       
    return dict1, dict2

def noiseRemoval(dict1, dict2=None):
    for key, value in dict1.items():
        value = hairRemoval(value)  # Hair removal step
        for c in range(value.shape[2]):
            value[:, :, c] = cv2.medianBlur(value[:, :, c], 3)
        dict1[key] = value
    
    if dict2:
        for key, value in dict2.items():
            value = hairRemoval(value)  # Hair removal step
            for c in range(value.shape[2]):
                value[:, :, c] = cv2.medianBlur(value[:, :, c], 3)
            dict2[key] = value 
        return dict1, dict2
    return dict1

def imageEnhancement(dict1, dict2=None):
    for key, value in dict1.items():
        for c in range(value.shape[2]):
            value[:, :, c] = np.uint8(cv2.normalize(value[:, :, c], None, 0, 255, cv2.NORM_MINMAX))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            value[:, :, c] = clahe.apply(value[:, :, c])
        dict1[key] = value
    
    if dict2:
        for key, value in dict2.items():
            for c in range(value.shape[2]):
                value[:, :, c] = np.uint8(cv2.normalize(value[:, :, c], None, 0, 255, cv2.NORM_MINMAX))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                value[:, :, c] = clahe.apply(value[:, :, c])
            dict2[key] = value
        return dict1, dict2
    return dict1

def segment(img):
    gray_image_blurred = cv2.GaussianBlur(img, (25, 25), 0)
    ret2, th2 = cv2.threshold(gray_image_blurred, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2

def segmentation(dict1, dict2=None):
    for key, value in dict1.items():
        for c in range(value.shape[2]):
            value[:, :, c] = segment(value[:, :, c])
        dict1[key] = value
    
    if dict2:
        for key, value in dict2.items():
            for c in range(value.shape[2]):
                value[:, :, c] = segment(value[:, :, c])
            dict2[key] = value
        return dict1, dict2
    return dict1
