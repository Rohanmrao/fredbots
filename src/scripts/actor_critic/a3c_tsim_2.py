import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_srvs.srv import Empty
import math

# A3C Network architecture
class A3CNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(A3CNetwork, self).__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.actor_logits = layers.Dense(num_actions)
        self.critic_value = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        logits = self.actor_logits(x)
        value = self.critic_value(x)
        return logits, value

# A3C Agent
class A3CAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.global_model = A3CNetwork(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.global_model(state)
        action_probabilities = tf.nn.softmax(logits)
        action = np.random.choice(self.num_actions, p=action_probabilities.numpy()[0])
        return action

    def compute_loss(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Compute logits and values
            logits, values = self.global_model(states)
            next_logits, _ = self.global_model(next_states)

            # Compute advantages and target values
            target_values = rewards + 0.99 * next_logits * (1 - dones)
            advantages = target_values - values

            # Compute actor and critic losses
            actions_one_hot = tf.one_hot(actions, self.num_actions)
            actor_loss = tf.reduce_sum(-tf.math.log_softmax(logits) * actions_one_hot, axis=1) * tf.stop_gradient(advantages)
            critic_loss = tf.square(target_values - values)

            # Total loss
            loss = tf.reduce_mean(actor_loss + critic_loss)

        return loss

    def train(self, envs, num_episodes):
        for episode in range(num_episodes):
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []

            for env in envs:
                state = env.reset()
                while True:
                    action = self.get_action(state)
                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)

                    episode_reward += reward

                    if done:
                        break

                    state = next_state

            loss = self.compute_loss(states, actions, rewards, next_states, dones)

            gradients = tape.gradient(loss, self.global_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.global_model.trainable_variables))

            print(f"Episode {episode+1}: Reward = {episode_reward}")


# TurtleSim Environment
class TurtleSimEnv:
    def __init__(self, turtle_name):
        self.turtle_name = turtle_name
        self.pose = Pose()
        self.pose_sub = rospy.Subscriber('/{}/pose'.format(self.turtle_name), Pose, self.pose_callback)
        self.cmd_vel_pub = rospy.Publisher('/{}/cmd_vel'.format(self.turtle_name), Twist, queue_size=10)
        rospy.sleep(1)  # Wait for publisher and subscriber to initialize

    def pose_callback(self, data):
        self.pose = data

    def get_pose(self):
        return self.pose

    def step(self, action):
        twist = Twist()
        if action == 0:  # Move forward
            twist.linear.x = 1.0
        elif action == 1:  # Move backward
            twist.linear.x = -1.0
        elif action == 2:  # Rotate clockwise
            twist.angular.z = -1.0
        elif action == 3:  # Rotate counterclockwise
            twist.angular.z = 1.0
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

# Create TurtleSim environments
turtle1_env = TurtleSimEnv('turtle1')
turtle2_env = TurtleSimEnv('turtle2')
turtle3_env = TurtleSimEnv('turtle3')
turtle4_env = TurtleSimEnv('turtle4')

# Create a list of TurtleSim environments
envs = [turtle1_env, turtle2_env, turtle3_env, turtle4_env]

# Use the `envs` list in the A3C agent or any other algorithm
