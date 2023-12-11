import pygame
import random
import cv2 
import pprint
import math
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
stack_size = 16

sky_y = 0  # Assuming the top of the frame is at y = 0
threshold_ground = 100  # Distance from ground to consider as 'close'
threshold_sky = 100  # Distance from sky to consider as 'close'

batch_size = 16
input_shape = (new_height, new_width, stack_size)
action_space = 2  
num_episodes = 1000 

epsilon = 0.7

large_positive_reward = 0.005
safety_distance = 50
new_safety_distance = int(safety_distance / downscale)
learning_rate = 0.00025
min_replay_memory_size = 16
training_batch_size = 16


def preprocess_frame(frame, new_width=new_width, new_height=new_height):
    """
    Preprocess the game frame for the DQN model.
    - Convert to grayscale (if RGB)
    - Resize
    - Normalize pixel values
    """
    # Convert to grayscale if it's an RGB image
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Proper conversion to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Normalize pixel values
    normalized_frame = resized_frame / 255.0

    # If the original frame was RGB, we need to expand the dimensions to match expected shape
    if frame.ndim == 3:
        normalized_frame = np.expand_dims(normalized_frame, axis=-1)

    return normalized_frame


def save_processed_state_as_png(state, info=None, filename='processed_state.png'):
    """
    Save the processed state (image frame) as a PNG file.
    
    :param state: The processed state (image frame).
    :param filename: Name of the file to save the image.
    """
    
    fig, ax = plt.subplots()
    ax.imshow(state)  # Assuming state is a grayscale image

    if info != None:
        bird_x = info['bird']['x'] / downscale
        bird_y = max(0, info['bird']['y'] / downscale)
        pipe = info['pipes'][0]  # Assuming the first pipe in the list is the next obstacle
        ground_y = info['base']['y'] / downscale
        # Calculate the center of the gap
        pipe_height = pipe['height'] / downscale
        pipe_bottom = pipe['bottom'] / downscale
        gap_center = (pipe_bottom + pipe_height) / 2
        gap_center_x = pipe['x'] / downscale
        pipe_x = pipe['x'] / downscale

        ax.plot([bird_x+64/8, pipe_x+100/8], [bird_y, gap_center], color='blue', linestyle='-', linewidth=2)
        ax.plot([pipe_x-500/8, pipe_x-64/8], [ground_y, gap_center+20/8], color='red', linestyle='-', linewidth=2)
        ax.plot([pipe_x-500/8, pipe_x-64/8], [sky_y, gap_center-20/8], color='red', linestyle='-', linewidth=2)

        ax.plot([pipe_x-64/8, pipe_x+100/8], [gap_center-60/8, gap_center-60/8], color='green', linestyle='-', linewidth=2)
        ax.plot([pipe_x-64/8, pipe_x+100/8], [gap_center+60/8, gap_center+60/8], color='green', linestyle='-', linewidth=2)
    
    plt.axis('off')  # Turn off axis numbers and labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# deprecated
def avoid_ground_sky_reward(state, action):
    '''
    Note: This reward function is not used in the final model.
    Give a large positive reward if the bird is close to the ground and the action is to jump.
    Give a large positive reward if the bird is close to the sky and the action is to not jump.

    :param state: The current state of the game.
    :param action: The action taken by the agent.
    :return: The reward for the action.
    '''
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

def pipe_reward(info):
    '''
    Calculates the reward for the agent based on the distance between the bird and the center of the gap.
    The reward is proportional to the euclidean distance between the bird and the center of the gap.
    The reward is 0 if the bird is not within the "safe zone", which is defined by the top and bottom lines, and the central zone of the gap.

    :param info: The current state of the game.
    :param action: The action taken by the agent.
    :return: The reward for the action.
    '''

    # Get the relevant information from the state
    bird_y = info['bird']['y']
    bird_x = info['bird']['x']
    ground_y = info['base']['y']
    pipe_info = info['pipes'][0] 
    pipe_x = pipe_info['x']
    pipe_height = pipe_info['height']
    pipe_bottom = pipe_info['bottom']
    gap_center_y = (pipe_bottom + pipe_height) / 2
    gap_center_x = pipe_x

    # calculate euclidean distance between bird and gap center
    euclidean_distance = np.sqrt((bird_y - gap_center_y)**2 + (bird_x - gap_center_x+100)**2)
    #print(f'Euclidean Distance: {euclidean_distance}')

    # calculate the top and bottom lines
    top_line_point1 = (pipe_x - 500, sky_y)
    top_line_point2 = (pipe_x-64, gap_center_y-20)
    bottom_line_point1 = (pipe_x - 500, ground_y)
    bottom_line_point2 = (pipe_x-64, gap_center_y+20)

    # check if bird is within the 'safe zone'
    above_top_line = is_point_above_line((bird_x, bird_y), top_line_point1, top_line_point2)
    above_bottom_line = is_point_above_line((bird_x, bird_y), bottom_line_point1, bottom_line_point2)
    in_pipe = bird_x > pipe_x - 64 and bird_x < pipe_x + 100
    pipe_center_threshold = bird_y > gap_center_y - 60 and bird_y < gap_center_y + 60

    # calculate reward based on bird's position
    if not above_top_line or above_bottom_line:
        reward = -0.001
    # elif in_pipe and not pipe_center_threshold:
    #     reward = -0.001
    else:
        # exponential reward function accentuates the difference between being close to the center and being far away
        reward = math.exp((410 - euclidean_distance) / 410)-1
    return max(-0.001, reward)

def instantiate_model(input_shape, action_space):
    '''
    Creates a new model with the specified input shape and action space.
    Compiles the model with the specified loss function and optimizer.

    :param input_shape: The shape of the input to the model.
    :param action_space: The number of actions the agent can take.
    :return: The compiled model.
    '''

    model = create_model(input_shape, action_space)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    return model

def process_state(state, info=None):
    '''
    Performs preprocessing of the frame and returns the processed state.
    If info is not None, the processed state is saved as a PNG file.

    :param state: The current state of the game.
    :param info: The info dictionary returned by the step function.
    :return: The processed state.
    '''
    if info is not None:
        state = preprocess_frame(state)  # Assuming the first element is the image
        save_processed_state_as_png(state, info, 'my_processed_frame.png')
    else:
        state = preprocess_frame(state[0])
        save_processed_state_as_png(state, info, 'my_processed_frame.png')
    state = np.reshape(state, (new_height, new_width, channels))
    return state

def randomize_action(action, epsilon=epsilon):
    '''
    Randomizes the action taken by the model with a probability of epsilon.
    The action is chosen randomly from the set of possible actions.

    :param action: The action chosen by the model.
    :param epsilon: The probability of randomizing the action.
    :return: The randomized action and a boolean indicating whether the action was randomized.
    '''
    randomize = random.random()
    randomized = False
    if randomize > epsilon:
        action = random.randint(0, 1)
        randomized = True
    return action, randomized

def get_reward(info, reward=0):
    '''
    Calculates the reward for the agent based on the information returned by the step function.

    :param info: The info dictionary returned by the step function.
    :param reward: The reward returned by the step function.
    :return: The reward for the action.
    '''

    #reward += avoid_ground_sky_reward(info, action)
    reward += pipe_reward(info) 
    return reward

def train_dqn(env, human_env=None):
    '''
    Trains the DQN model using the specified environment.

    :param env: The environment to train the model on.
    :param human_env: The environment to render the model on.
    '''

    # Initialize the model and replay memory
    model = instantiate_model(input_shape, action_space)
    replay_memory = []
    large_reward_memory_1 = []
    large_reward_memory_0 = []
    epsilon = 0.7

    # Training loop for each episode
    for episode in range(num_episodes):
        
        # Reset the environment
        state_raw = env.reset()
        # Process the initial state
        state = process_state(state_raw)
        # Assuming state is the initial frame with shape [100, 72, 1]
        frame_stack = np.repeat(state, stack_size, axis=-1)
        # Assuming frame_stack is the initial state with shape [100, 72, stack_size]
        state_batch = np.expand_dims(frame_stack, axis=0)

        # Initialize the episode variables
        total_reward = 0
        count = 0

        while True:
            count += 1
            # predict the action
            action = np.argmax(model.predict(state_batch, verbose=0)[0])  
            # randomize the action with probability epsilon
            action, randomized = randomize_action(action, epsilon=epsilon)

            # Take the action and get the next state
            next_state_raw, reward, done, _, info = env.step(action)
            # Process the next state
            next_state = process_state(next_state_raw, info)

            # Remove the oldest frame and add the new frame
            new_frame_stack = np.append(frame_stack[:, :, 1:], next_state, axis=-1)  
            # Get the custom reward        
            reward = get_reward(info, reward)
            # Update the total reward
            total_reward += reward

            print(f'Action: {action}, Reward: {reward}, Randomize: {randomized}')

            # Add the experience to the replay memory, and the large reward memory if applicable
            replay_memory.append((frame_stack, action, reward, new_frame_stack, done, info))
            if reward > 0.85 and action == 1:
                large_reward_memory_1.append((frame_stack, action, reward, new_frame_stack, done, info))
            if reward > 0.85 and action == 0:
                large_reward_memory_0.append((frame_stack, action, reward, new_frame_stack, done, info))

            # Update the frame stack
            frame_stack = new_frame_stack
            state_batch = np.expand_dims(frame_stack, axis=0)

            if done:
                break
        
        # Model training from the last episode
        print('Training Replay...')
        # Sample a minibatch from the replay memory
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

        # Fine tuning with the large reward memory
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
        if len(large_reward_memory_1) > 100:
            large_reward_memory_1 = replay_memory[-100:]
        
        if len(large_reward_memory_0) > 100:
            large_reward_memory_0 = replay_memory[-100:]
        
        # Implement randomize threshold decay with episode
        if epsilon < 1:
            epsilon += 0.001
        
        if episode % 10 == 0:
            # Save the model every 10 episodes
            model.save('flappy_bird_dqn.keras')

            # Optionally, render the model, check how it performs
            # This is useful when the training is done with render_mode='rgb_array'
            # human_env = gym.make('FlappyBird-v0', render_mode='human')
            # state_raw = human_env.reset()
            # state = process_state(state_raw)
            # frame_stack = np.repeat(state, stack_size, axis=-1)
            
            # while True:
            #     action = np.argmax(model.predict(np.expand_dims(frame_stack, axis=0), verbose=0)[0])
            #     next_state_raw, reward, done, _, info = human_env.step(action)
            #     next_state = process_state(next_state_raw, info)
            #     new_frame_stack = np.append(frame_stack[:, :, 1:], next_state, axis=-1)
            #     frame_stack = new_frame_stack
            #     if done:
            #         break
            # human_env.close()
        
        print(f'Episode: {episode}, Total Reward: {total_reward}, Randomize Threshold: {epsilon}')

def main():
    env = gym.make('FlappyBird-v0', render_mode='human')
    train_dqn(env)
    env.close()

if __name__ == '__main__':
    main()
