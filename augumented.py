import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Path to save augmented images
augmented_path = 'C:/verynewprojectfile/inputs/augmented'

# Create the directory if it doesn't exist
os.makedirs(augmented_path, exist_ok=True)

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_and_save_images(image_path, save_path, num_augmentations=5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Generate augmented images
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        augmented_image = batch[0].astype('uint8')
        augmented_image = augmented_image.squeeze()  # Remove unnecessary dimensions
        augmented_filename = f"{os.path.basename(image_path).split('.')[0]}_aug_{i}.png"
        cv2.imwrite(os.path.join(save_path, augmented_filename), augmented_image)
        i += 1
        if i >= num_augmentations:
            break

# Example usage
image_path = 'C:/verynewprojectfile/inputs/train/benign/melanoma_13.jpg'  # Replace with your image path
augment_and_save_images(image_path, augmented_path)
