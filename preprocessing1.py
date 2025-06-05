import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayScaleConversion(path):
    dict1 = {}
    dict2 = {}
    classes = os.listdir(path)

    for classe in classes:
        pathFile = os.path.join(path, classe)
        images = os.listdir(pathFile)

        for image in images:
            imagePath = os.path.join(pathFile, image)
            img = cv2.imread(imagePath)
            
            if img is not None:
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if classe == 'benign':
                    dict1[imagePath] = imgGray
                else:
                    dict2[imagePath] = imgGray
            else:
                print(f"Error: Unable to read image {imagePath}")

    return dict1, dict2

def dullRazor(dict1, dict2):
    def apply_dullrazor(img):
        if len(img.shape) > 2:
            grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayScale = img
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
        ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    
        return dst

    for key, value in dict1.items():
        dict1[key] = apply_dullrazor(value)
    
    for key, value in dict2.items():
        dict2[key] = apply_dullrazor(value)
    
    return dict1, dict2

def noiseRemoval(dict1, dict2):
    for key, value in dict1.items():
        new_img = cv2.medianBlur(value, 3)
        dict1[key] = new_img
    
    for key, value in dict2.items():
        new_img = cv2.medianBlur(value, 3)
        dict2[key] = new_img 
    
    return dict1, dict2

def imageEnhancement(dict1, dict2):
    for key, value in dict1.items():
        value = np.uint8(cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        value = clahe.apply(value)
        dict1[key] = value
    
    for key, value in dict2.items():
        value = np.uint8(cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        value = clahe.apply(value)
        dict2[key] = value
    
    return dict1, dict2

def segment(img):
    gray_image_blurred = cv2.GaussianBlur(img, (25, 25), 0)
    ret2, th2 = cv2.threshold(gray_image_blurred, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2

def segmentation(dict1, dict2):
    for key, value in dict1.items():
        newValue = segment(value)
        dict1[key] = newValue
    for key, value in dict2.items():
        newValue = segment(value)
        dict2[key] = newValue
    return dict1, dict2

if __name__ == "__main__":
    path_train = r'D:\verynewprojectfile\inputs\train'
    dict1, dict2 = grayScaleConversion(path_train)
    
    # Apply DullRazor to all images in dict1 and dict2
    dict1, dict2 = dullRazor(dict1, dict2)

    dict1, dict2 = noiseRemoval(dict1, dict2)
    dict1, dict2 = imageEnhancement(dict1, dict2)
    dict1, dict2 = segmentation(dict1, dict2)

    # Example to display an image
    img1 = dict1[r'D:\verynewprojectfile\inputs\train\benign\melanoma_13.jpg']
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.show()
