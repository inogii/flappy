
import pygame
import random
import cv2 
import pprint
import flappy_bird_env  # noqa
import numpy as np
import gymnasium as gym
import tensorflow as tf
from collections import deque
from model import create_model
import matplotlib.pyplot as plt


global gamma
gamma = 0.99
global new_width 
new_width = 80
global new_height
new_height = 80


def preprocess_frame(frame, new_width=80, new_height=80):
    """
    Preprocess the game frame for the DQN model.
    - Check if conversion to grayscale is needed
    - Resize
    - Normalize pixel values
    """
    # Check if frame is already grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:  # RGB image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Normalize pixel values
    normalized_frame = resized_frame / 255.0

    return normalized_frame




def train_dqn(env):
    # Define hyperparameters
    input_shape = (80, 80, 1)  # Example: downsized image with 4-frame stack
    action_space = 2  # Flap or not
    num_episodes = 1000 # Adjust as needed
    batch_size = 128  # Adjust as needed
    # Create the DQN model
    model = create_model(input_shape, action_space)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001))
    replay_memory = []
    # Training loop
    for episode in range(num_episodes):

        state_raw = env.reset()
        state = preprocess_frame(state_raw[0])  # Assuming the first element is the image
        state = state.reshape(1, new_width, new_height, 1)  # Add batch and channel dimensions
        
        total_reward = 0

        while True:
            action = np.argmax(model.predict(state))  # Predict the action
            if episode < 100:
                if random.random() < 0.5:
                    action = random.randint(0, 1)
            next_state_raw, reward, done, _, info = env.step(action)
            next_state = preprocess_frame(next_state_raw[0])  # Assuming the first element is the image
            next_state = next_state.reshape(1, new_width, new_height, 1)  # Add batch and channel dimensions

            # Store in replay memory
            replay_memory.append((state, action, reward, next_state, done))
    
            state = next_state
            total_reward += reward

            if done:
                break

        # Sample a batch from replay memory and train the model (you need to implement this)

        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            for state_batch, action, reward, next_state_batch, done in minibatch:
                target = reward
                if not done:
                    target = reward + gamma * np.amax(model.predict(next_state_batch))
                target_f = model.predict(state_batch)
                target_f[0][action] = target
                model.fit(state_batch, target_f, epochs=1, verbose=0)
                print(f'Episode: {episode}, Total Reward: {total_reward}')
            replay_memory = []

# Initialize the Flappy Bird environment
env = gym.make('FlappyBird-v0', render_mode='human')

# Train the DQN
train_dqn(env)

# Close the environment
env.close()
