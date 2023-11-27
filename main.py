
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
new_width = 72
global new_height
new_height = 100


def preprocess_frame(frame, new_width=72, new_height=100):
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

def save_processed_state_as_png(state, filename='processed_state.png'):
    """
    Save the processed state (image frame) as a PNG file.
    
    :param state: The processed state (image frame).
    :param filename: Name of the file to save the image.
    """
    plt.imshow(state, cmap='gray')  # Assuming state is a grayscale image
    plt.axis('off')  # Turn off axis numbers and labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def avoid_ground_sky_reward(state, action):
    bird_y = state['bird']['y']
    ground_y = state['base']['y']
    sky_y = 0  # Assuming the top of the frame is at y = 0

    # Define thresholds for proximity to the ground and sky
    threshold_ground = 100  # Distance from ground to consider as 'close'
    threshold_sky = 100  # Distance from sky to consider as 'close'

    large_positive_reward = 1  # Example value

    # Reward for avoiding the ground
    if ground_y - bird_y < threshold_ground and action == 1:
        reward = large_positive_reward
    elif ground_y - bird_y < threshold_ground and action == 0:
        reward = -large_positive_reward
    # Reward for avoiding the sky
    elif bird_y - sky_y < threshold_sky and action == 0:
        reward = large_positive_reward
    elif bird_y - sky_y < threshold_sky and action == 1:
        reward = -large_positive_reward
    else:
        reward = 0  # Default reward

    return reward



def train_dqn(env):
    # Define hyperparameters
    input_shape = (new_height, new_width, 1)  # Example: downsized image with 4-frame stack
    action_space = 2  # Flap or not
    num_episodes = 1000 # Adjust as needed
    batch_size = 512  # Adjust as needed
    # Create the DQN model
    model = create_model(input_shape, action_space)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001))
    replay_memory = []
    # Training loop
    for episode in range(num_episodes):

        state_raw = env.reset()
        state = preprocess_frame(state_raw[0])  # Assuming the first element is the image
        save_processed_state_as_png(state, 'my_processed_frame.png')
        state = state.reshape(1, new_height, new_width, 1)  # Add batch and channel dimensions
        total_reward = 0
        count = 0

        while True:
            action = np.argmax(model.predict(state, verbose=0))  # Predict the action
            randomize = random.random() * (num_episodes - episode) / num_episodes
            if randomize > 0.5:
                action = random.randint(0, 1)
            next_state_raw, reward, done, _, info = env.step(action)
            next_state = preprocess_frame(next_state_raw)  # Assuming the first element is the image
            next_state = next_state.reshape(1, new_height, new_width, 1)  # Add batch and channel dimensions

            reward += avoid_ground_sky_reward(info, action)  # Custom reward function
            print(f'Action: {action}, Reward: {reward}, Randomize: {randomize>0.5}')

            # Store in replay memory
            replay_memory.append((state, action, reward, next_state, done))
    
            state = next_state
            total_reward += reward

            if done:
                break

        # Sample a batch from replay memory and train the model (you need to implement this)

        if len(replay_memory) > 4*batch_size:
            replay_memory = random.sample(replay_memory, 4*batch_size)
            print('Training model...')
            for state_batch, action, reward, next_state_batch, done in replay_memory:
                
                target = reward
                # if asdf < 10:
                #     asdf += 1
                #     save_processed_state_as_png(state_batch[0], f'my_processed_frame_{asdf}.png')
                # if not done:
                #     target = reward + gamma * np.argmax(model.predict(next_state_batch, verbose=0))
                target_f = model.predict(state_batch, verbose=0)
                target_f[0][action] = target
                model.fit(state_batch, target_f, epochs=5, verbose=0)
            print(f'Episode: {episode}, Total Reward: {total_reward}')

# Initialize the Flappy Bird environment
env = gym.make('FlappyBird-v0', render_mode='human')

# Train the DQN
train_dqn(env)

# Close the environment
env.close()
