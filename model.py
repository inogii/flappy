import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization

def create_model(input_shape, action_space):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(16, (6, 6), strides=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Conv2D(64, (4, 4), strides=(1, 1)))
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Conv2D(128, (2, 2), strides=(1, 1)))
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(tf.keras.layers.Activation('relu'))

    model.add(Flatten())  # Flattening the layer before fully connected layers

    # Fully connected layers
    model.add(Dense(128))
    model.add(BatchNormalization())  # Batch normalization layer
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.add(Dense(action_space, activation='sigmoid'))  # Output layer

    return model

