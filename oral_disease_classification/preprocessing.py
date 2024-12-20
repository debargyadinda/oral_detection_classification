import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to datasets
TRAIN_DIR = '/home/tatsuhirosatou/proj/oral_disease_classification/dataset/TRAIN/'
TEST_DIR = '/home/tatsuhirosatou/proj/oral_disease_classification/dataset/TEST/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation and Preprocessing
def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, test_generator

if __name__ == "__main__":
    train_gen, test_gen = create_generators()
    print("Train and Test Generators Created")