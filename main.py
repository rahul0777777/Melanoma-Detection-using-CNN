import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
from hairRemoval import dullRazor_single
from preprocessing1 import grayScaleConversion, dullRazor, noiseRemoval, imageEnhancement, segmentation, segment
import matplotlib.pyplot as plt


# Function to plot images
def plot_images(images, titles, cmap=None):
    n = len(images)
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Define paths
path_train = 'D:/verynewprojectfile/inputs/train'
path_test = 'D:/verynewprojectfile/inputs/test'
augmented_path = 'D:/verynewprojectfile/inputs/augmented'
os.makedirs(augmented_path, exist_ok=True)

# Function to load and preprocess images
def load_and_preprocess_images(path):
    dict1, dict2 = grayScaleConversion(path)
    dict1, dict2 = dullRazor(dict1, dict2)
    dict1, dict2 = noiseRemoval(dict1, dict2)
    dict1, dict2 = imageEnhancement(dict1, dict2)
    dict1, dict2 = segmentation(dict1, dict2)
    return dict1, dict2

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

# Function to augment and save images
def augment_and_save_images(image, base_filename, save_path, num_augmentations=5):
    img = np.expand_dims(image, axis=0)  # Add batch dimension
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        augmented_image = batch[0].astype('uint8')
        augmented_image = augmented_image.squeeze()  # Remove unnecessary dimensions
        augmented_filename = f"{base_filename}_aug_{i}.png"
        cv2.imwrite(os.path.join(save_path, augmented_filename), augmented_image)
        i += 1
        if i >= num_augmentations:
            break

# Function to preprocess a single image and display the steps
def preprocess_and_display_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return

    titles = ['Original', 'Grayscale', 'DullRazor', 'Noise Removed', 'Enhanced', 'Segmented']
    images = [img]

    # Grayscale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img_gray)

    # DullRazor
    img_dullrazor = dullRazor_single(img_gray)
    images.append(img_dullrazor)

    # Noise removal
    img_noise_removed = cv2.medianBlur(img_dullrazor, 3)
    images.append(img_noise_removed)

    # Image enhancement
    img_enhanced = np.uint8(cv2.normalize(img_noise_removed, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_enhanced)
    images.append(img_enhanced)

    # Segmentation
    img_segmented = segment(img_enhanced)
    images.append(img_segmented)

    # Plot the images
    plot_images(images, titles, cmap='gray')

image_path = 'D:/verynewprojectfile/inputs/train/benign/melanoma_13.jpg'  # Replace with the path to the image you want to check
preprocess_and_display_image(image_path)

# Preprocess training and testing images
dict1, dict2 = load_and_preprocess_images(path_train)
dict_test1, dict_test2 = load_and_preprocess_images(path_test)

# Function to create datasets
def create_dataset(dict1, dict2, target_size=(224, 224)):
    df = pd.DataFrame(columns=['images', 'outcome'])
    for key, value in dict1.items():
        img = cv2.resize(value, target_size)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
        df.loc[len(df.index)] = [img, 'benign']
        augment_and_save_images(img, key, os.path.join(augmented_path, 'benign'))
    for key, value in dict2.items():
        img = cv2.resize(value, target_size)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
        df.loc[len(df.index)] = [img, 'malignant']
        augment_and_save_images(img, key, os.path.join(augmented_path, 'malignant'))
    return df

# Create datasets
df_train = create_dataset(dict1, dict2)
df_test = create_dataset(dict_test1, dict_test2)

# Prepare data for training
x_train = np.array(df_train['images'].tolist()).astype('float32') / 255.0
y_train = df_train['outcome'].map({"benign": 0, "malignant": 1}).to_numpy()
y_train = to_categorical(y_train, num_classes=2)

x_test = np.array(df_test['images'].tolist()).astype('float32') / 255.0
y_test = df_test['outcome'].map({"benign": 0, "malignant": 1}).to_numpy()
y_test = to_categorical(y_test, num_classes=2)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Define the CNN model
cnn = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[224, 224, 1]),
    Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(0.25),  # Add dropout layer

    Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(0.25),  # Add dropout layer

    Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    MaxPool2D(2),
    Dropout(0.25),  # Add dropout layer

    Flatten(),
    Dense(256, activation='relu', name='feature_dense1'),
    Dropout(0.5),  # Add dropout layer
    Dense(128, activation='relu', name='feature_dense2'),
    Dropout(0.5),  # Add dropout layer
    Dense(64, activation='relu', name='feature_dense3'),
    Dropout(0.5),  # Add dropout layer
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn.summary()

optimizer = Adamax()

cnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = cnn.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_test, y_test), shuffle=True, callbacks=[early_stopping])

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show(block=False)  # Non-blocking
plt.pause(3)

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show(block=False)  # Non-blocking
plt.pause(3)

# Save the model
cnn.save('cnn.h5')

# Extract features from the feature_dense3 layer
feature_layer_model = Model(inputs=cnn.inputs, outputs=cnn.get_layer(name="feature_dense3").output)

# Extract features for training and testing data
features_train = feature_layer_model.predict(x_train)
features_test = feature_layer_model.predict(x_test)

# Convert features and labels to DataFrames
df_features_train = pd.DataFrame(features_train)
df_features_train['target'] = df_train['outcome'].map({"benign": 0, "malignant": 1}).to_numpy()

df_features_test = pd.DataFrame(features_test)
df_features_test['target'] = df_test['outcome'].map({"benign": 0, "malignant": 1}).to_numpy()

# Save features to Excel file
with pd.ExcelWriter('cnn_features.xlsx') as writer:
    df_features_train.to_excel(writer, sheet_name='Train Features', index=False)
    df_features_test.to_excel(writer, sheet_name='Test Features', index=False)

# Example usage of preprocess_and_display_image
preprocess_and_display_image(image_path)
