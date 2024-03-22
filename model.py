import numpy as np
import os
import cv2
import keras
import tensorflow as tf
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


train_path = './archive/train'
test_path = './archive/test'

folder_list_train = os.listdir(train_path)
folder_list_train.sort()

folder_list_test = os.listdir(test_path)
folder_list_test.sort()


# create empty arrays for training:
x_train = []
y_train = []

# create empty arrays for testing:
x_test = []
y_test = []

# Load the training data into the array:
for i, category in enumerate(folder_list_train):
    files = os.listdir(train_path + '/' + category)

    for file in files:
        img = cv2.imread(train_path+'/'+category+'/{0}'.format(file),0)
        x_train.append(img)
        y_train.append(i)

# Load the test data into the array:
for i, category in enumerate(folder_list_test):
    files = os.listdir(test_path + '/' + category)

    for file in files:
        img = cv2.imread(test_path+'/'+category+'/{0}'.format(file),0)
        x_test.append(img)
        y_test.append(i)


# Convert the arrays to numpy arrays:
x_train = np.array(x_train, "float32")
y_train = np.array(y_train, "float32")
x_test = np.array(x_test, "float32")
y_test = np.array(y_test, "float32")


# Normalize the data to the range of 0 to 1:
x_train = x_train/255
x_test = x_test/255

# Reshape the training data:
number_of_images = x_train.shape[0]
x_train = x_train.reshape(number_of_images, 48, 48, 1)

# Reshape the test data:
number_of_images = x_test.shape[0]
x_test = x_test.reshape(number_of_images, 48, 48, 1)

# Convert the labels to categorical:
y_train = utils.to_categorical(y_train, 7)
y_test = utils.to_categorical(y_test, 7)



# BUILD THE MODEL

input_shape = x_train.shape[1:]
print(f"Input shape: {input_shape}")

model = Sequential()

# Layers
# 1st convolutional layer
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # strides means that the movement along the convolutional layer will be in 2 positions

# 2nd convolutional layer
model.add(Conv2D(input_shape=input_shape, filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # strides means that the movement along the convolutional layer will be in 2 positions

# 3rd convolutional layer
model.add(Conv2D(input_shape=input_shape, filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # strides means that the movement along the convolutional layer will be in 2 positions

# 4th convolutional layer
model.add(Conv2D(input_shape=input_shape, filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # strides means that the movement along the convolutional layer will be in 2 positions

# Flatten the result:
model.add(Flatten())

# Fully connected layer:
model.add(Dense(units=4096, activation="relu"))

# Dropout layer:
model.add(Dropout(0.5))

# Fully connected layers:
model.add(Dense(4096, activation="relu"))
model.add(Dense(7,activation="softmax")) # 7 categories

print(f"Model summary: {model.summary()}")

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

batch=32 # number of images to be trained pro batch, decrease if you run out of memory
epochs=30

stepsPerEpoch = np.ceil(len(x_train)/batch)
validationSteps = np.ceil(len(x_test)/batch)

stopEarly = EarlyStopping(monitor='val_accuracy' , patience=5) # patience is the number of epochs with no improvement after which training will be stopped.



# TRAIN THE MODEL

history = model.fit(x_train, y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    shuffle=True,
                    callbacks=[stopEarly])

modelFileName = os.path.join(os.getcwd(), 'model_emotion.keras')
model.save(modelFileName)



# PLOT THE RESULTS
def plot_model_performance(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # show train and validation train chart
    plt.plot(epochs, acc , 'r' , label="Train accuracy")
    plt.plot(epochs, val_acc , 'b' , label="Validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Training and validation Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # show loss and validation loss chart
    plt.plot(epochs, loss , 'r' , label="Train loss")
    plt.plot(epochs, val_loss , 'b' , label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training and validation Loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
