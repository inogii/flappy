import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def create_model(input_shape, action_space):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

    model.add(Flatten())  # Flattening the layer before fully connected layers

    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space, activation='linear'))  # Output layer

    return model
