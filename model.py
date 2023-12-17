import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Input
from keras.applications import VGG16

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

# def create_model(input_shape, action_space):

#     base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

#     # Freeze the layers of the base model to prevent them from being updated during training
#     for layer in base_model.layers:
#         layer.trainable = False

#     # Add new layers for our specific problem
#     x = Flatten()(base_model.output)
#     x = Dense(1024, activation='relu')(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dense(action_space, activation='sigmoid')(x)  # Change to your specific output

#     # Create the new model
#     model = Model(inputs=base_model.input, outputs=x)

#     return model

