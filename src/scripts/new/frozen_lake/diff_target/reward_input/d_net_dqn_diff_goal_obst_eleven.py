import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# print gpu info
print(tf.config.list_physical_devices('GPU'))

# Create a neural network model
def create_model(input_shape, num_actions):
    model = Sequential([
        Input(shape=input_shape),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Create a class for the environment
class Env():
    def __init__(self, grid_size=11, max_steps=500):
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        self.obstacles = np.array([[2,0],[2,1],[2,2],[2,3],[2,4],[6,6],[6,7],[6,8],[6,9],[6,10]])
        # self.rewards = np.zeros((grid_size, grid_size))
        self.reset()
        self.reset_goal()
        self.immediate_reward = 0

    def check(self, test, array):
        return any(np.array_equal(x, test) for x in array)

    def reset(self):
        self.pos = np.random.randint(0, self.grid_size, size=2)
        while self.pos in self.obstacles:
            self.pos = np.random.randint(0, self.grid_size, size=2)
        self.steps = 0
        self.done = False
        return self.pos
    
    def reset_goal(self):
        self.goal = np.random.randint(0, self.grid_size, size=2)
        while self.goal in self.obstacles:
            self.goal = np.random.randint(0, self.grid_size, size=2)
        print('Goal:', self.goal)
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         self.rewards[i, j] = -self.euclidean_distance_from_goal(np.array([i, j]))
        # self.rewards[self.goal[0], self.goal[1]] = 100
        return self.goal
    
    # def step(self, action):
    #     self.steps += 1
    #     if action == 0 and self.pos[0] < self.grid_size - 1: # right
    #         self.pos[0] += 1
    #     elif action == 1 and self.pos[0] > 0: # left
    #         self.pos[0] -= 1
    #     elif action == 2 and self.pos[1] > 0: # down
    #         self.pos[1] -= 1
    #     elif action == 3 and self.pos[1] < self.grid_size - 1: # up
    #         self.pos[1] += 1
    #     else:
    #         pass
    #         # raise ValueError('Invalid action')
    #     if np.array_equal(self.pos, self.goal):
    #         self.done = True
    #         reward = 0
    #         # reward = 100
    #     elif self.steps >= self.max_steps:
    #         self.done = True
    #         reward = self.rewards[self.pos[0], self.pos[1]]
    #     else:
    #         reward = self.rewards[self.pos[0], self.pos[1]]
    #     return self.pos, reward, self.done

    def step(self, action): # As per the paper
        self.steps += 1
        prev_pos = self.pos.copy()
        if action == 0 and self.pos[0] < self.grid_size - 1 and not (self.pos + np.array([1, 0])) in self.obstacles: # right
            self.pos[0] += 1
        elif action == 1 and self.pos[0] > 0 and not (self.pos - np.array([1, 0])) in self.obstacles: # left
            self.pos[0] -= 1
        elif action == 2 and self.pos[1] > 0 and not (self.pos - np.array([0, 1])) in self.obstacles: # down
            self.pos[1] -= 1
        elif action == 3 and self.pos[1] < self.grid_size - 1 and not (self.pos + np.array([0, 1])) in self.obstacles: # up
            self.pos[1] += 1
        else:
            reward = -150
            self.immediate_reward = reward
            self.done = False
            return self.pos, reward, self.done, True # TODO: The episode is not terminated.
        if np.array_equal(self.pos, self.goal):
            self.done = True
            reward = 500
        elif self.steps >= self.max_steps:
            self.done = True
            reward = 0
        else:
            if self.euclidean_distance_from_goal(self.pos) < self.euclidean_distance_from_goal(prev_pos):
                reward = 10
            else:
                reward = -10
        self.immediate_reward = reward
        return self.pos, reward, self.done, False
        

    def euclidean_distance_from_goal(self, pos):
        dist = np.sqrt(np.sum((pos - self.goal) ** 2))
        return dist
    
# Create an agent class
class Agent():
    def __init__(self, env, model, target_model):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.target_model.set_weights(self.model.get_weights())
        self.gamma = 0.7
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        # self.epsilon_decay = 0.001
        self.epsilon_decay = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        

    def add_to_memory(self, state, goal, action, reward, next_state, done):
        self.memory.append((state, goal, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)
        else:
            # Pre-process the input
            inputt = np.concatenate((state, self.env.goal, np.array([self.env.immediate_reward])))
            inputt = tf.convert_to_tensor(inputt)
            inputt = tf.expand_dims(inputt, 0)

            return np.argmax(self.model.predict(inputt, verbose=0)[0]) # TODO: check the predict output

    def predict(self, inputt):

        return np.argmax(self.model.predict(inputt, verbose=0)[0])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, goal, action, reward, next_state, done in batch:
            
            target = reward

            if not done:
                # Pre-process the next state input
                input_next = np.concatenate((next_state, goal, np.array([reward])))
                input_next = tf.convert_to_tensor(input_next)
                input_next = tf.expand_dims(input_next, 0)

                target += self.gamma * np.amax(self.target_model.predict(input_next, verbose=0)[0])

            # Pre-process the current state input
            inputt = np.concatenate((state, goal, np.array([reward])))
            inputt = tf.convert_to_tensor(inputt)
            inputt = tf.expand_dims(inputt, 0)

            cur_q_value = self.model.predict(inputt, verbose=0) # Q-value of current state
            cur_q_value[0][action] = target # TODO: check the predict output
            
            self.model.fit(inputt, cur_q_value, epochs=1, verbose=0)
            
# Initialize the agent

try:
    model = tf.keras.models.load_model('dqn_with_reward.h5')
    print("Loaded model from disk")
    agent = Agent(Env(), model=model, target_model=model)
except:
    print("Creating new model")
    agent = Agent(Env(), model=create_model(input_shape=(5,), num_actions=4), target_model=create_model(input_shape=(5,), num_actions=4))

# Load the memory
try:
    filee = open('memory.bin', 'rb')
    agent.memory = pickle.load(filee)
    filee.close()
    print("Loaded memory from disk")
except:
    print("Creating new memory")

# Load the starting episode number
try:
    filee = open('num_episodes.bin', 'rb')
    start_ep = pickle.load(filee)
    filee.close()
    print("Loaded num_episodes from disk")
except:
    print("Creating new num_episodes")
    start_ep = 0

# Train the agent

num_episodes = 100
reward_lst = []

file = open('rewards.txt', 'a')

for episode in range(start_ep, start_ep+num_episodes):
    state_lst = [] # DEBUG
    ep_reward = 0 # DEBUG
    state = agent.env.reset()
    for step in range(agent.env.max_steps):
        state_lst.append(state.copy()) # DEBUG
        # print('State:', state) # DEBUG
        # print('state_lst:', state_lst) # DEBUG
        action = agent.act(state)
        # print('Action:', action) # DEBUG
        next_state, reward, done, terminate = agent.env.step(action)
        ep_reward += reward # DEBUG
        # print(f"next_state: {next_state}, reward: {reward}, done: {done}") # DEBUG
        # next_state = np.reshape(next_state, [1, 2])
        agent.add_to_memory(state, agent.env.goal, action, reward, next_state, done)
        state = next_state
        if done:
            if np.array_equal(agent.env.goal, agent.env.pos): # Reached the goal
                agent.env.reset_goal()
        elif (agent.env.steps >= agent.env.max_steps) or terminate:
            print('Episode: {}/{}, steps: {}, e: {:.2}'.format(episode, start_ep+num_episodes, step+1, agent.epsilon))
            print('State list:', state_lst) # DEBUG
            break
    if len(agent.memory) > agent.batch_size:
        agent.replay()
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon = (1 - agent.epsilon_min) * np.exp(-agent.epsilon_decay*episode) + agent.epsilon_min
    if episode % 5 == 0:
        agent.model.save('dqn_with_reward.h5')    
    agent.target_model.set_weights(agent.model.get_weights())

    reward_lst.append(ep_reward) # DEBUG
    file.write(f"{episode},{ep_reward}\n") # DEBUG
    file.flush() # DEBUG

agent.model.save('dqn_with_reward.h5')

print('Average reward:', np.mean(reward_lst))

# Save the memory
filee = open('memory.bin', 'wb')
pickle.dump(agent.memory, filee)
filee.close()

# Save the episode number
filee = open('num_episodes.bin', 'wb')
pickle.dump(episode+1, filee)
filee.close()

file.close() # DEBUG

