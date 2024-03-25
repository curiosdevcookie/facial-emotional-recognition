import numpy as np
import os
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
import os

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    labels = []
    for label, category in enumerate(sorted(os.listdir(folder))):
        category_path = os.path.join(folder, category)
        for file in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize images to 48x48
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image: {img_path}, error: {e}")
    return np.array(images, dtype="float32"), np.array(labels, dtype="float32")

train_path = './archive/train'
test_path = './archive/test'

x_train, y_train = load_images_from_folder(train_path)
x_test, y_test = load_images_from_folder(test_path)

# Normalize and reshape
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 7)
y_test = keras.utils.to_categorical(y_test, 7)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Model architecture
model = Sequential([
  Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(48, 48, 1)),
  BatchNormalization(),
  MaxPooling2D(2, 2),
  Dropout(0.25),
  Conv2D(64, (3, 3), padding="same", activation="relu"),
  BatchNormalization(),
  MaxPooling2D(2, 2),
  Dropout(0.25),
  Conv2D(128, (3, 3), padding="same", activation="relu"),
  BatchNormalization(),
  MaxPooling2D(2, 2),
  Dropout(0.25),
  Flatten(),
  Dense(512, activation='relu'),
  BatchNormalization(),
  Dropout(0.5),
  Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Callbacks
stopEarly = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

batch = 64
epochs = 60

# TRAIN THE MODEL

history = model.fit(datagen.flow(x_train, y_train, batch_size=batch),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[stopEarly])

modelFileName = os.path.join(os.getcwd(), 'model_emotion.keras')
model.save(modelFileName)



# PLOT THE RESULTS
def plot_model_performance(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'r-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'r-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()

plot_model_performance(history)