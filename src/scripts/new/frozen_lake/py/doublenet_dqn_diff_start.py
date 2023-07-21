# Deep Q Network for navigating through a grid world
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# print gpu info
print(tf.config.list_physical_devices('GPU'))
time.sleep(5)

start_time = time.time()

# Create a neural network model
def create_model(input_shape, num_actions):
    model = Sequential([
        Input(shape=input_shape),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.005))
    return model

# Create a class for the environment
class Env():
    def __init__(self, grid_size=6, max_steps=30):
        self.grid_size = grid_size
        self.max_steps = max_steps
        # self.goal = np.random.randint(0, grid_size, size=2) # random goal
        self.goal = np.array([4, 3]) # fixed goal
        print('Goal:', self.goal)
        self.reset()

    def reset(self):
        self.pos = np.random.randint(0, self.grid_size, size=2)
        self.steps = 0
        self.done = False
        return self.pos
    
    def step(self, action):
        prev_pos = self.pos.copy()
        self.steps += 1
        if action == 0: # up
            self.pos[0] = min(self.pos[0] + 1, self.grid_size - 1)
        elif action == 1: # down
            self.pos[0] = max(self.pos[0] - 1, 0)
        elif action == 2: # left
            self.pos[1] = max(self.pos[1] - 1, 0)
        elif action == 3: # right
            self.pos[1] = min(self.pos[1] + 1, self.grid_size - 1)
        else:
            raise ValueError('Invalid action')
        if np.array_equal(self.pos, self.goal):
            self.done = True
            reward = 100
        elif self.steps >= self.max_steps:
            self.done = True
            reward = 0
        else:
            if self.euclidean_distance_from_goal(self.pos) < self.euclidean_distance_from_goal(prev_pos):
                reward = 1
            else:
                reward = -1
        return self.pos, reward, self.done

    def euclidean_distance_from_goal(self, pos):
        dist = np.sqrt(np.sum((pos - self.goal) ** 2))
        return dist
        
# Create an agent class
class Agent():
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.gamma = 0.5
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.1
        self.batch_size = 32
        self.memory = deque(maxlen=1000)
        self.target_model = create_model((1, 2), 4)
        self.target_model.set_weights(self.model.get_weights())

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)
        else:
            # Pre-process the state
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            return np.argmax(self.model.predict(state, verbose=0)[0][0]) # TODO: check the predict output

    def predict(self, state):
        # Pre-process the state
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        return np.argmax(self.model.predict(state, verbose=0)[0][0])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            
            target = reward

            if not done:
                # Pre-process the next state
                next_state = tf.convert_to_tensor(next_state)
                next_state = tf.expand_dims(next_state, 0)

                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])

            # Pre-process the state
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            cur_q_value = self.model.predict(state, verbose=0) # Q-value of current state
            cur_q_value[0][0][action] = target
            
            self.model.fit(state, cur_q_value, epochs=1, verbose=0)
            

# Initialize the agent

try:
    model = tf.keras.models.load_model('doublenet_dqn_diff_start.h5')
    print("Loaded model from disk")
    agent = Agent(Env(), model=model)
except:
    print("Creating new model")
    agent = Agent(Env(), model=create_model(input_shape=(1, 2), num_actions=4))


# Train the agent

num_episodes = 25
reward_lst = []

for episode in range(num_episodes):
    state_lst = [] # DEBUG
    ep_reward = 0 # DEBUG
    state = agent.env.reset()
    state = np.reshape(state, [1, 2])
    for step in range(agent.env.max_steps):
        state_lst.append(state.copy()) # DEBUG
        # print('State:', state) # DEBUG
        # print('state_lst:', state_lst) # DEBUG
        action = agent.act(state)
        # print('Action:', action) # DEBUG
        next_state, reward, done = agent.env.step(action)
        ep_reward += reward # DEBUG
        # print(f"next_state: {next_state}, reward: {reward}, done: {done}") # DEBUG
        next_state = np.reshape(next_state, [1, 2])
        agent.add_to_memory(state, action, reward, next_state, done)
        state = next_state
        if done:
            print('Episode: {}/{}, steps: {}, e: {:.2}'.format(episode, num_episodes, step+1, agent.epsilon))
            print('State list:', state_lst) # DEBUG
            break
        if len(agent.memory) > agent.batch_size:
            agent.replay()
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon = (1 - agent.epsilon_min) * np.exp(-agent.epsilon_decay*episode) + agent.epsilon_min
    if episode % 5 == 0:    
        agent.target_model.set_weights(agent.model.get_weights())
        agent.model.save('doublenet_dqn_diff_start.h5')
    reward_lst.append(ep_reward)

print('Average reward:', np.mean(reward_lst))

# Save the model
agent.model.save('doublenet_dqn_diff_start.h5')

time_taken = time.time() - start_time
print(f"Time taken: {time_taken:.2f} seconds")