import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# A3C Network architecture
class A3CNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(A3CNetwork, self).__init__()
        self.dense1 = layers.Dense(24, activation='relu')
        self.dense2 = layers.Dense(24, activation='relu')
        self.policy_logits = layers.Dense(num_actions)
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


'''
logits are the unnormalized predictions or scores produced by 
the model before applying a softmax activation function. 
Logits represent the raw, unprocessed outputs of the 
last layer of a neural network, typically used 
for multiclass classification problems.
'''

# Actor Network architecture
# class ActorNetwork(tf.keras.Model):
#     def __init__(self, num_actions):
#         super(ActorNetwork, self).__init__()
#         self.dense1 = layers.Dense(24, activation='relu')
#         self.dense2 = layers.Dense(24, activation='relu')
#         self.policy_logits = layers.Dense(num_actions)

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         logits = self.policy_logits(x)
#         return logits

# # Critic Network architecture
# class CriticNetwork(tf.keras.Model):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()
#         self.dense1 = layers.Dense(24, activation='relu')
#         self.dense2 = layers.Dense(24, activation='relu')
#         self.value = layers.Dense(1)

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         value = self.value(x)
#         return value
    


# A3C Agent
class A3CAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.global_model = A3CNetwork(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def loss(self, states, actions, rewards, next_states, dones):
        logits, values = self.global_model(states)
        _, next_values = self.global_model(next_states)

        # Compute advantages and discounted rewards
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = discount_rewards(deltas, gamma * lambda_)
        discounted_rewards = discount_rewards(rewards, gamma)

        # Compute policy loss
        actions_one_hot = tf.one_hot(actions, self.num_actions)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantages)
        policy_loss = tf.reduce_mean(policy_loss)

        # Compute value loss
        value_loss = tf.reduce_mean(tf.square(deltas))

        # Total loss
        total_loss = policy_loss + value_loss

        return total_loss

    def train(self, envs, max_episodes):
        for episode in range(max_episodes):
            states, actions, rewards, next_states, dones = self.run_episode(envs)

            with tf.GradientTape() as tape:
                total_loss = self.loss(states, actions, rewards, next_states, dones)

            grads = tape.gradient(total_loss, self.global_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))

    def run_episode(self, envs):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for step in range(max_steps):
            # Collect states from all environments
            current_states = []
            for env in envs:
                print("env.get_pose(): ",env.get_pose())
                current_states.append(env.get_pose())

            # Convert states to numpy array
            states_np = np.array(current_states)

            # states_tensor = tf.convert_to_tensor([states_np], dtype=tf.float32)

            # Predict actions and values
            logits, values = self.global_model(states_np)

            # Sample actions from policy logits
            actions_np = tf.random.categorical(logits, num_samples=1).numpy().flatten()

            # Execute actions in all environments
            for i, env in enumerate(envs):
                action = actions_np[i]
                next_state, reward, done = env.step(action)

                # Collect data
                states.append(current_states[i])
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                # Update state
                current_states[i] = next_state

        return states, actions, rewards, next_states,dones

# Helper function to discount rewards
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    cumulative_reward = 0

    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        discounted_rewards[i] = cumulative_reward

    return discounted_rewards

# ROS TurtleSim Environment
class TurtleSimEnv:
    def __init__(self, turtle_name):
        self.turtle_name = turtle_name
        self.pose = Pose()
        self.pose_sub = rospy.Subscriber('/turtle1/pose'.format(self.turtle_name), Pose, self.pose_callback)
        self.cmd_vel_pub = rospy.Publisher('/turtle1/cmd_vel'.format(self.turtle_name), Twist, queue_size=10)
        rospy.sleep(1)  # Wait for publisher and subscriber to initialize

    def pose_callback(self, data):
        # self.pose = data
        self.state = [
            data.x,
            data.y,
            data.theta,
            data.linear_velocity,
            data.angular_velocity,
        ]

    def get_pose(self):
        return self.state


    def step(self, action):
        twist = Twist()
        twist.linear.x = action  # Use action as linear velocity
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.1)  # Wait for the turtle to move
        next_state = self.get_pose()
        reward = self.calculate_reward(next_state)
        done = self.check_done(next_state)
        return next_state, reward, done

    def calculate_reward(self, state):
        # Define your reward function based on the state
        return 0  # Placeholder, implement your own

    def check_done(self, state):
        # Define your termination condition based on the state
        return False  # Placeholder, implement your own

    def reset(self):
        # Reset the environment and return the initial state
        # Implement the reset logic for your specific environment
        return self.get_pose()

# ROS Initialization
rospy.init_node('a3c_turtlesim')

# Hyperparameters
num_actions = 3
gamma = 0.99
lambda_ = 0.95
max_episodes = 1000
max_steps = 200

# Create multiple TurtleSim environments
envs = [TurtleSimEnv('turtle1'), TurtleSimEnv('turtle2'), TurtleSimEnv('turtle3'), TurtleSimEnv('turtle4')]

# Create A3C agent and train
agent = A3CAgent(num_actions)
agent.train(envs, max_episodes)
