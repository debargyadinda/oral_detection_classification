import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
MODEL_PATH = '/home/tatsuhirosatou/proj/oral_disease_classification/oral_disease_model.keras'
TEST_DATA_PATH = '/home/tatsuhirosatou/proj/oral_disease_classification/dataset/TEST/'

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Predict
y_pred = model.predict(test_gen)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# Generate Confusion Matrix
cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))

# Plotting Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_true, np.argmax(y_pred, axis=1), target_names=class_labels)
print(report)
