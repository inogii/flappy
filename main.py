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

# Define hyperparameters
gamma = 0.99
width = 576
height = 800
downscale = 8
new_width = int(width/downscale)
new_height = int(height/downscale)
channels = 1
stack_size = 4

sky_y = 0  # Assuming the top of the frame is at y = 0
threshold_ground = 200  # Distance from ground to consider as 'close'
threshold_sky = 200  # Distance from sky to consider as 'close'

batch_size = 16
input_shape = (new_height, new_width, stack_size)
action_space = 2  
num_episodes = 1000 

randomize_threshold = 0.7

large_positive_reward = 0.001
safety_distance = 50
new_safety_distance = int(safety_distance / downscale)
learning_rate = 0.001



def preprocess_frame(frame, new_width=new_width, new_height=new_height):
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

def save_processed_state_as_png(state, info=None, filename='processed_state.png'):
    """
    Save the processed state (image frame) as a PNG file.
    
    :param state: The processed state (image frame).
    :param filename: Name of the file to save the image.
    """
    
    fig, ax = plt.subplots()
    ax.imshow(state, cmap='gray')  # Assuming state is a grayscale image

    if info != None:
        bird_x = info['bird']['x'] / downscale
        bird_y = max(0, info['bird']['y'] / downscale)
        pipe = info['pipes'][0]  # Assuming the first pipe in the list is the next obstacle

        # Calculate the center of the gap
        pipe_height = pipe['height'] / downscale
        pipe_bottom = pipe['bottom'] / downscale
        gap_center = (pipe_bottom + pipe_height) / 2
        
        ax.axhline(pipe_height-new_safety_distance, color='black', linewidth=4)
        ax.axhline(pipe_bottom+new_safety_distance, color='black', linewidth=4)
        ax.axhline(gap_center, color='white', linewidth=2)

        ax.plot([bird_x, bird_x], [min(bird_y, gap_center), max(bird_y, gap_center)], color='blue', linestyle='-', linewidth=2)

    plt.axis('off')  # Turn off axis numbers and labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def avoid_ground_sky_reward(state, action):
    bird_y = state['bird']['y']
    ground_y = state['base']['y']

    # Reward for avoiding the ground
    if ground_y - bird_y < threshold_ground and action == 1:
        reward = large_positive_reward
    # Reward for avoiding the sky
    elif bird_y - sky_y < threshold_sky and action == 0:
        reward = large_positive_reward
    else:
        reward = 0  # Default reward

    return reward

def pipe_reward(info, action):
    bird_y = info['bird']['y']
    pipe_info = info['pipes'][0]  # Assuming the first pipe is the next obstacle
    pipe_x = pipe_info['x']
    pipe_height = pipe_info['height']
    pipe_bottom = pipe_info['bottom']
    gap_center = (pipe_bottom + pipe_height) / 2

    reward = 0

    # Check if the pipe is visible on the screen
    if 0 < pipe_x <= width:
        # The bird is within the vertical range of the pipe
        if bird_y > pipe_height-safety_distance and bird_y < pipe_bottom+safety_distance:
            # Calculate the distance from the bird to the center of the gap
            distance_center_bird = bird_y - gap_center
            # Define the maximum possible distance (half the gap height)
            max_distance = (pipe_bottom - pipe_height) / 2
            if distance_center_bird > 0 and action == 1:
                # Bird is below the center of the gap
                reward = large_positive_reward * abs(distance_center_bird) / max_distance
            elif distance_center_bird < 0 and action == 0:
                # Bird is above the center of the gap
                reward = large_positive_reward * abs(distance_center_bird) / max_distance

    return reward

def instantiate_model(input_shape, action_space):
    model = create_model(input_shape, action_space)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001))
    return model

def process_state(state, info=None):
        if info is not None:
            state = preprocess_frame(state)  # Assuming the first element is the image
            save_processed_state_as_png(state, info, 'my_processed_frame.png')
        else:
            state = preprocess_frame(state[0])
            save_processed_state_as_png(state, info, 'my_processed_frame.png')
        state = np.reshape(state, (new_height, new_width, channels))
        return state

def randomize_action(action):
    randomize = random.random()
    randomized = False
    if randomize > randomize_threshold:
        action = random.randint(0, 1)
        randomized = True
    return action, randomized

def get_reward(info, action):
    reward = 0
    reward += avoid_ground_sky_reward(info, action)
    reward += pipe_reward(info, action) 
    return reward

def train_dqn(env):
    model = instantiate_model(input_shape, action_space)
    replay_memory = []
    large_reward_memory = []

    for episode in range(num_episodes):
        state_raw = env.reset()
        
        state = process_state(state_raw)
        
        # Assuming state is the initial frame with shape [100, 72, 1]
        frame_stack = np.repeat(state, stack_size, axis=-1)
        print(frame_stack.shape)
        state_batch = np.expand_dims(frame_stack, axis=0)
        print(state_batch.shape)

        total_reward = 0

        while True:
            # Prepare stacked_state for prediction
            action = np.argmax(model.predict(state_batch, verbose=0)[0])  # Predict the action
            action, randomized = randomize_action(action)

            next_state_raw, reward, done, _, info = env.step(action)
            next_state = process_state(next_state_raw, info)

            # Remove the oldest frame and add the new frame
            frame_stack = np.append(frame_stack[:, :, 1:], next_state, axis=-1)          
            reward = get_reward(info, action)
            print(f'Action: {action}, Reward: {reward}, Randomize: {randomized}')

            replay_memory.append((frame_stack, action, reward, next_state, done, info))
            if reward > 0:
                large_reward_memory.append((state, action, reward, next_state, done, info))

            state_batch = np.expand_dims(frame_stack, axis=0)
            total_reward += reward

            if done:
                break


        print('Training model...')
        # if total_reward > 0:
        #     for state_batch, action, reward, next_state_batch, done, info in replay_memory:
        #         target = reward
        #         if not done:
        #             # Compute the target for non-terminal states
        #             target += gamma * np.amax(model.predict(next_state_batch, verbose=0))

        #         # Predict the target Q-values for the current state
        #         target_f = model.predict(state_batch, verbose=0)

        #         # Update the target for the action taken
        #         target_f[0][action] = target

        #         # Fit the model
        #         model.fit(state_batch, target_f, epochs=1, verbose=0)
        
       # Ensure large_reward_memory has enough samples
       # Ensure large_reward_memory has enough samples
        # Assuming large_reward_memory is available
        if len(large_reward_memory) >= 10*batch_size:
            # Sample a batch from large_reward_memory
            reward_sample = random.sample(large_reward_memory, batch_size)

            # Initialize arrays for batch training
            state_batch = np.array([sample[0] for sample in reward_sample])

            # Compute targets for batch
            target_batch = np.zeros((batch_size, action_space)) # Initialize target batch with the correct shape
            for i, (state, action, reward, next_state, done, info) in enumerate(reward_sample):
                target = reward
                if not done:
                    # Reshape next_state for single sample prediction
                    next_state_reshaped = np.expand_dims(next_state, axis=0)
                    target += gamma * np.amax(model.predict(next_state_reshaped, verbose=0))
                # Update the target for the action taken
                target_batch[i][action] = target

            # Fit the model on the entire batch
            model.fit(state_batch, target_batch, epochs=20, verbose=1)

            # Prune large_reward_memory to keep the best 1000 rewards
            large_reward_memory = sorted(large_reward_memory, key=lambda x: x[2], reverse=True)[:1000]





def main():
    env = gym.make('FlappyBird-v0', render_mode='human')
    train_dqn(env)
    env.close()

if __name__ == '__main__':
    main()
