# Reinforcement Learning for Flappy Bird

## Introduction
This project is an exploration of applying reinforcement learning (RL) techniques to the popular game Flappy Bird. By leveraging the capabilities of modern machine learning libraries and game development tools, we aim to train an intelligent agent capable of mastering Flappy Bird.

## Motivation
The intersection of gaming and artificial intelligence (AI) offers a rich ground for experimentation and learning. Flappy Bird, with its simple yet challenging gameplay, provides an ideal environment for implementing and testing RL algorithms. This project not only demonstrates the practical application of RL but also serves as an engaging way to visualize and understand the capabilities of AI in a gaming context.

## Results

In the video below, you can see the performance of the model during the training process, as well as some of the most important milestones: passing the first and second pipe.

https://github.com/inogii/flappy_rl/assets/94673757/3fd7b622-3f74-4fdf-a934-c8fef170bbf1

## Approach
### Reinforcement Learning Framework
- We employ the `gymnasium` (formerly known as `gym`) framework to create a structured environment where our RL agent can learn and interact.
- The core of our RL approach involves training an agent to make decisions based on game states, aiming to maximize its score in the game.

## Model and Training

### Image Preprocessing
- Utilizing `cv2`, we preprocess game frames for efficient learning.
- This involves resizing and downscaling the images, making them more manageable for the model to process without losing essential details.
- The preprocessing steps are crucial for reducing computational overhead while maintaining the quality of inputs for the model.
- We used a downscaling factor of 8, one grayscale channel, and a frame stacking of 8 frames to keep track of the movement of the bird

### Model Architecture
Our model is a Sequential model built using TensorFlow and Keras, designed specifically for processing image data. This architecture is particularly suited for the Flappy Bird game, where understanding visual cues is crucial for making decisions. The model comprises the following layers:

- **Convolutional Layers**: 
  - The first layer has 16 filters with a kernel size of 6x6, ideal for capturing low-level features from the game frames.
  - This is followed by a 64-filter layer with 4x4 kernels, and a 128-filter layer with 2x2 kernels, progressively extracting more complex features.
  - Each convolutional layer is accompanied by batch normalization, enhancing model stability and reducing overfitting.

- **Activation Functions**: 
  - `ReLU` (Rectified Linear Unit) is used in the convolutional layers, providing non-linearity and helping the model learn complex patterns in the data.
  - The final dense layer uses a `Sigmoid` activation, suitable for binary decisions like 'flap' or 'not flap'.

- **Flattening and Dense Layers**: 
  - Flattening is applied twice, transforming the output of the convolutional layers into a format suitable for dense layers.
  - Dense layers with 128, 1024, and 1024 neurons respectively, provide the capability to make decisions based on the extracted features.

### Why This Model?
- **Image Processing Capability**: The convolutional layers make this model highly effective for processing and interpreting the visual information from Flappy Bird, a key aspect for the agent's performance.
- **Feature Extraction**: The progressive increase in filters allows the model to recognize a wide range of features, from basic shapes to more complex patterns, crucial for understanding different game scenarios.
- **Decision Making**: The dense layers, especially with the sigmoid activation in the final layer, enable the agent to make discrete decisions, vital for gameplay in Flappy Bird.

### Training Process
- The agent undergoes training sessions where it interacts with the game environment, learning from the outcomes of its actions.
- The model is trained to predict the best possible action (flap or not) based on the current state of the game, using rewards as feedback.
- Continuous training and optimization aim to maximize the agent's score and its ability to navigate through the game effectively.

### Reward System

We explored several different reward systems to make our model learn better from the input. Both of them involved scaling the rewards according to the distance to the pipe, the fewer the distance, the higher the reward for the agent would be. 

In this first video, you can see the vector that goes from the bird to the pipe, every time the length of this vector decreases, a reward is given to the agent.



https://github.com/inogii/flappy_rl/assets/94673757/b869170f-a87e-487a-990e-c1fb210ff03b


In the second video, you can see the approach involving safe zones, where rewards proportional to the distance would be only given to the agent when the bird stayed in the safe zones.



https://github.com/inogii/flappy_rl/assets/94673757/79edc0c3-1af9-4a7b-9c84-a8dfe191c946



## Conclusion
This project stands as a testament to the power of reinforcement learning in game environments. It showcases the potential of AI to learn and excel in tasks that are intuitive for humans but challenging for machines.

## Future Work
- Enhancing the model for better performance and efficiency.
- Exploring more complex RL algorithms and architectures.
