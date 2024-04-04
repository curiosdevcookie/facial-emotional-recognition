import numpy as np
import os
import cv2
from keras.layers import RandomZoom
from keras import utils
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    labels = []
    for label, category in enumerate(sorted(os.listdir(folder))):
        if category.startswith('.'):
            continue
        print(f"Assigning label {label} to category {category}")
        categorylabels_path = os.path.join(folder, category)
        if os.path.isdir(categorylabels_path):
            print(f"Processing folder from filepath: {categorylabels_path} : {category}")
            for file in os.listdir(categorylabels_path):
                img_path = os.path.join(categorylabels_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize images to 48x48
                    images.append(img)
                    labels.append(label)
    return np.array(images, dtype="float32"), np.array(labels, dtype="float32")

train_path = './archive/train'
test_path = './archive/test'

images_train, labels_train = load_images_from_folder(train_path)
images_test, labels_test = load_images_from_folder(test_path)


print("Unique labels in training set:", np.unique(labels_train))
print("Unique labels in test set:", np.unique(labels_test))

# Normalize and reshape
images_train, images_test = images_train / 255.0, images_test / 255.0
images_train = images_train.reshape(-1, 48, 48, 1)
images_test = images_test.reshape(-1, 48, 48, 1)

print("Labels in training set before conversion:", labels_train)
print("Labels in test set before conversion:", labels_test)

labels_train = labels_train.astype(int)
labels_test = labels_test.astype(int)

# Convert labels to categorical
labels_train = utils.to_categorical(labels_train, 8)
labels_test = utils.to_categorical(labels_test, 8)


# Model architecture
model = Sequential([
    Input(shape=(48, 48, 1)),
    RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="reflect", interpolation="bilinear"),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
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