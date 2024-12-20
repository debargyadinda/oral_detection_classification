import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from preprocessing import create_generators

# Constants
IMG_SIZE = (224, 224, 3)
NUM_CLASSES = len(create_generators()[0].class_indices)  # Dynamically get the number of classes

# Build Model
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Train Model
def train_model():
    train_gen, test_gen = create_generators()

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'oral_disease_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )

    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Model Trained and Saved")
