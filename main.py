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
stack_size = 8

sky_y = 0  # Assuming the top of the frame is at y = 0
threshold_ground = 100  # Distance from ground to consider as 'close'
threshold_sky = 100  # Distance from sky to consider as 'close'

batch_size = 16
input_shape = (new_height, new_width, stack_size)
action_space = 2  
num_episodes = 1000 

randomize_threshold = 0.8

large_positive_reward = 0.005
safety_distance = 50
new_safety_distance = int(safety_distance / downscale)
learning_rate = 0.00025
min_replay_memory_size = 16
training_batch_size = 16



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
        ground_y = info['base']['y'] / downscale
        # Calculate the center of the gap
        pipe_height = pipe['height'] / downscale
        pipe_bottom = pipe['bottom'] / downscale
        gap_center = (pipe_bottom + pipe_height) / 2
        pipe_x = pipe['x'] / downscale

        ax.plot([bird_x, pipe_x], [min(bird_y, gap_center), max(bird_y, gap_center)], color='blue', linestyle='-', linewidth=2)
        ax.plot([bird_x, pipe_x], [ground_y, pipe_bottom], color='red', linestyle='-', linewidth=2)
        ax.plot([bird_x, pipe_x], [sky_y, pipe_height], color='red', linestyle='-', linewidth=2)
    
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
    # elif ground_y - bird_y < threshold_ground and action == 0:
    #     reward = -large_positive_reward
    # elif bird_y - sky_y < threshold_sky and action == 1:
    #     reward = -large_positive_reward
    else:
        reward = 0  # Default reward

    return reward

def is_point_above_line(point, line_point1, line_point2):
    """
    Check if a point is above or below the line defined by two points.
    
    :param point: Tuple (x, y) for the point to check.
    :param line_point1: Tuple (x, y) for the first point on the line.
    :param line_point2: Tuple (x, y) for the second point on the line.
    :return: True if the point is above the line, False if below.
    """
    # Calculate the slope (m)
    m = (line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0])
    # Calculate the y-intercept (b)
    b = line_point1[1] - m * line_point1[0]
    
    # Calculate the y value of the line at the x position of the point
    y_line_at_point_x = m * point[0] + b
    
    # If the y value of the point is greater than the line's y value, it's above the line
    return point[1] > y_line_at_point_x

def pipe_reward(info, action):
    bird_y = info['bird']['y']
    bird_x = info['bird']['x']
    ground_y = info['base']['y']
    pipe_info = info['pipes'][0]  # Assuming the first pipe is the next obstacle
    pipe_x = pipe_info['x']
    pipe_height = pipe_info['height']
    pipe_bottom = pipe_info['bottom']
    gap_center_y = (pipe_bottom + pipe_height) / 2
    gap_center_x = pipe_x
    # calculate euclidean distance between bird and gap center
    euclidean_distance = np.sqrt((bird_y - gap_center_y)**2 + (0 - gap_center_x)**2)
    top_line_point1 = (pipe_x - 500, sky_y)
    top_line_point2 = (pipe_x, pipe_height)
    bottom_line_point1 = (pipe_x - 500, ground_y)
    bottom_line_point2 = (pipe_x, pipe_bottom)
    # check if bird is above or below the top line
    above_top_line = is_point_above_line((bird_x, bird_y), top_line_point1, top_line_point2)
    # check if bird is above or below the bottom line
    above_bottom_line = is_point_above_line((bird_x, bird_y), bottom_line_point1, bottom_line_point2)
    if not above_top_line or above_bottom_line:
        reward = -0.001
    else:
        reward = (1000 - euclidean_distance) / 1000
    return reward

def instantiate_model(input_shape, action_space):
    model = create_model(input_shape, action_space)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01), metrics=['accuracy'])
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

def randomize_action(action, randomize_threshold=randomize_threshold):
    randomize = random.random()
    randomized = False
    if randomize > randomize_threshold:
        action = random.randint(0, 1)
        randomized = True
    return action, randomized

def get_reward(info, action, reward=0):
    #reward += avoid_ground_sky_reward(info, action)
    reward += pipe_reward(info, action) 
    return reward

def train_dqn(env):
    model = instantiate_model(input_shape, action_space)
    replay_memory = []
    large_reward_memory_1 = []
    large_reward_memory_0 = []
    randomize_threshold = 0.7

    for episode in range(num_episodes):
        state_raw = env.reset()
        
        state = process_state(state_raw)
        
        # Assuming state is the initial frame with shape [100, 72, 1]
        frame_stack = np.repeat(state, stack_size, axis=-1)
        #print(frame_stack.shape)
        state_batch = np.expand_dims(frame_stack, axis=0)
        #print(state_batch.shape)

        total_reward = 0
        count = 0
        while True:
            count += 1
            # Prepare stacked_state for prediction
            action = np.argmax(model.predict(state_batch, verbose=0)[0])  # Predict the action
            action, randomized = randomize_action(action, randomize_threshold=randomize_threshold)

            next_state_raw, reward, done, _, info = env.step(action)
            next_state = process_state(next_state_raw, info)

            # Remove the oldest frame and add the new frame
            new_frame_stack = np.append(frame_stack[:, :, 1:], next_state, axis=-1)          
            reward = get_reward(info, action, reward)
            total_reward += reward
            print(f'Action: {action}, Reward: {reward}, Randomize: {randomized}')

            replay_memory.append((frame_stack, action, reward, new_frame_stack, done, info))
            if reward > 0.5 and action == 1:
                large_reward_memory_1.append((frame_stack, action, reward, new_frame_stack, done, info))
            if reward > 0.5 and action == 0:
                large_reward_memory_0.append((frame_stack, action, reward, new_frame_stack, done, info))

            frame_stack = new_frame_stack
            state_batch = np.expand_dims(frame_stack, axis=0)

            if done:
                break
        
        print('Training Replay...')
        current_states = np.array([experience[0] for experience in replay_memory])
        actions = np.array([experience[1] for experience in replay_memory])
        rewards = np.array([experience[2] for experience in replay_memory])
        next_states = np.array([experience[3] for experience in replay_memory])
        dones = np.array([experience[4] for experience in replay_memory])

        # Predict Q-values for current and next states
        current_q_values = model.predict(current_states, verbose=0)
        next_q_values = model.predict(next_states, verbose=0)

        # Compute target Q-values
        target_q_values = rewards + gamma * np.amax(next_q_values, axis=1) * (~dones)

        # Update the Q-values for the actions taken
        target_q_values_full = current_q_values
        for i, action in enumerate(actions):
            target_q_values_full[i][action] = target_q_values[i]

        # Train the model
        model.fit(current_states, target_q_values_full, batch_size=training_batch_size, epochs=1, verbose=0)
        replay_memory = []

        # Training loop
        if len(large_reward_memory_1) > min_replay_memory_size and len(large_reward_memory_0) > min_replay_memory_size:
            print('Fine tuning...')
            minibatch_1 = random.sample(large_reward_memory_1, training_batch_size)
            minibatch_0 = random.sample(large_reward_memory_0, training_batch_size)
            minibatch = minibatch_1 + minibatch_0

            # Extracting components of the experiences
            current_states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])

            # Predict Q-values for current and next states
            current_q_values = model.predict(current_states, verbose=0)
            next_q_values = model.predict(next_states, verbose=0)

            # Compute target Q-values
            target_q_values = rewards + gamma * np.amax(next_q_values, axis=1) * (~dones)

            # Update the Q-values for the actions taken
            target_q_values_full = current_q_values
            for i, action in enumerate(actions):
                target_q_values_full[i][action] = target_q_values[i]

            # Train the model
            model.fit(current_states, target_q_values_full, batch_size=training_batch_size, epochs=3, verbose=0)

        # Optionally, prune the replay memory if it gets too large
        if len(large_reward_memory_1) > 20000:
            large_reward_memory_1 = replay_memory[-20000:]
        
        if len(large_reward_memory_0) > 20000:
            large_reward_memory_0 = replay_memory[-20000:]
        
        # implement randomize threshold decay with episode
        if randomize_threshold < 1:
            randomize_threshold += 0.001
        
        if episode % 10 == 0:
            model.save('flappy_bird_dqn.keras')
            print(f'Episode: {episode}, Total Reward: {total_reward}, Randomize Threshold: {randomize_threshold}')

def main():
    env = gym.make('FlappyBird-v0', render_mode='human')
    train_dqn(env)
    env.close()

if __name__ == '__main__':
    main()
