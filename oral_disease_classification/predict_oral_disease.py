import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the path to your model and test data
MODEL_PATH = '/home/tatsuhirosatou/proj/oral_disease_classification/oral_disease_model.keras'
TEST_DATA_PATH = '/home/tatsuhirosatou/proj/oral_disease_classification/dataset/TEST/'

# Import os module
import os

# Load the trained model using TFSMLayer
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Assuming model expects 224x224 input
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input

    # Make prediction
    prediction = model(img_array)
    return prediction

# Example usage with a random image
random_image_path = os.path.join(TEST_DATA_PATH, 'example_image.jpg')  # Update with a valid image path
prediction = predict_image(random_image_path)

# Mapping prediction to class labels
class_labels = ['Caries', 'Gingivitis']
predicted_class = class_labels[np.argmax(prediction)]

print(f"Prediction for {random_image_path}: {predicted_class}")
