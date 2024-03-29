import numpy as np
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from keras.layers import RandomCrop, RandomFlip, RandomRotation, RandomZoom
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
        categorlabels_path = os.path.join(folder, category)
        for file in os.listdir(categorlabels_path):
            try:
                img_path = os.path.join(categorlabels_path, file)
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

images_train, labels_train = load_images_from_folder(train_path)
images_test, labels_test = load_images_from_folder(test_path)

# Normalize and reshape
images_train, images_test = images_train / 255.0, images_test / 255.0
images_train = images_train.reshape(-1, 48, 48, 1)
images_test = images_test.reshape(-1, 48, 48, 1)

# Convert labels to categorical
labels_train = utils.to_categorical(labels_train, 7)
labels_test = utils.to_categorical(labels_test, 7)


# Model architecture
model = Sequential([
    RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="reflect", interpolation="bilinear"),
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Callbacks
stopEarly = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

epochs = 30
batch_size = 64

# TRAIN THE MODEL
history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, validation_data=(images_test, labels_test), callbacks=[stopEarly])


modelFileName = os.path.join(os.getcwd(), 'output/model_emotion.keras')
model.save(modelFileName)


# TEST THE MODEL

test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)
print('\nTest accuracy:', test_acc)


# PLOT THE RESULTS
def plot_model_performance(history):
    # This is the model's accuracy on the training dataset. Accuracy is a measure of how well the model's predictions match the actual labels.
    acc = history.history['accuracy']
    # This is the model's accuracy on the validation dataset. The validation accuracy is a good indicator of how well the model is generalizing to unseen data.
    val_acc = history.history['val_accuracy']
    # The loss function measures how well the model's predictions match the actual labels.
    # A higher loss value indicates that the model's predictions are less accurate.
    loss = history.history['loss']
    # This is the value of the loss function for the validation dataset.
    # The validation loss is a good indicator of how well the model is generalizing to unseen data.
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