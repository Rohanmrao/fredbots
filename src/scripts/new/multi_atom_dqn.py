import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from math import atan2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the Q-Network model
def create_q_network(input_shape, action_space):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.model = create_q_network(state_shape, action_space)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize the ROS node
rospy.init_node('multiagent_dqn')

# Define the number of agents
num_agents = 2

# Initialize the action space and state shape
action_space = 4  # Example action space size, adjust according to your actual actions
state_shape = (360,)

# Initialize the DQN agents
agents = []
for _ in range(num_agents):
    agents.append(DQNAgent(state_shape, action_space))

# Define the callback function for the LaserScan messages
def laser_scan_callback(data, agent_index):
    lidar_ranges = np.array(data.ranges)
    state = np.reshape(lidar_ranges, state_shape)
    action = agents[agent_index].act(state)
    # Publish the action (Twist message) to control the agent

# Create the LaserScan subscribers for each agent
laser_subs = []
for i in range(num_agents):
    laser_subs.append(rospy.Subscriber(f'/agent{i+1}/scan', LaserScan, laser_scan_callback, i))

# Create the publishers for each agent's actions
action_pubs = []
for i in range(num_agents):
    action_pubs.append(rospy.Publisher(f'/agent{i+1}/cmd_vel', Twist, queue_size=1))

# Training loop
num_episodes = 1000  # Set the number of training episodes
batch_size = 32  # Set the batch size for replay
for episode in range(num_episodes):
    # Reset the environment and agents
    reset_sim = rospy.ServiceProxy('/reset', Empty)
    reset_sim()
    for agent in agents:
        agent.memory = []

    done = False
    while not done:
        # Collect state, action, reward, and next_state for each agent
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(num_agents):
            # Collect state, action, reward, and next_state for agent i
            # Use laser_scan_callback to get state and action
            # Calculate the reward based on the agent's position or task-specific criteria
            # Use laser_scan_callback to get next_state
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        # Store the experience in the agents' memories
        for i in range(num_agents):
            agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)

        # Perform replay for each agent
        for i in range(num_agents):
            agents[i].replay(batch_size)

# Use the trained agents to control the agents in the turtlesim environment
while not rospy.is_shutdown():
    rospy.spin()
