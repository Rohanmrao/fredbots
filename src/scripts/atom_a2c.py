#!/usr/bin/env python3

import math
import numpy as np
import rospy
import gym
from gym import spaces

import tensorflow as tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tensorflow import keras
from tensorflow.keras import layers
import tf.transformations as tf_trans

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


# Actor model
class Actor(keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.policy_logits = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        logits = self.policy_logits(x)
        return logits


# Critic model
class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        values = self.values(x)
        return values


# Actor-Critic model
class ActorCriticModel(keras.Model):
    def __init__(self, num_actions):
        super(ActorCriticModel, self).__init__()
        self.actor = Actor(num_actions)
        self.critic = Critic()

    def call(self, inputs):
        logits = self.actor(inputs)
        values = self.critic(inputs)
        return logits, values


# Custom Actor-Critic agent for turtlesim
class TurtlesimActorCriticAgent:
    def __init__(self, num_actions):
        self.model = ActorCriticModel(num_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # Huber loss is a type of loss function commonly used in regression tasks, including reinforcement learning.

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        print("shape of state to get_action fn: ", state.shape)
        logits, _ = self.model(state)  # state is the input
        # print("logits: ", logits)
        # print("self.model.output[0] for actions: ",self.model.output[0])
        action_probabilities = tf.nn.softmax(logits)
        # print("action_probs b4 converting to numpy: ",action_probabilities)
        action_probabilities = action_probabilities[0].numpy()  # Convert to NumPy array
        # print("action probability after converting to numpy ", action_probabilities)
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        print("chosen action ", action)
        return action

    def train(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:  # tf.GradientTape() is a TensorFlow API that enables automatic differentiation.
            # states = tf.convert_to_tensor([states], dtype=tf.float32)
            print("shape of states to train fn: ", states.shape)

            # states = self.state
            # states = tf.convert_to_tensor([states], dtype=tf.float32)

            # print("self.model.output[0]: ",self.model.output[0])

            logits, values = self.model(states)
            print("logits: ", logits)
            advantage = discounted_rewards - values
            print("ACTIONS SHAPE BEFORE ONE HOT")
            print(actions.shape)
            # print("self.model.output[0]",self.model.output[0])
            # actions_one_hot = tf.one_hot(actions, depth=self.model.output[0])
            # print(actions_one_hot,"%%"*5)
            # print("self.model.output[0]: ",self.model.output[0])

            # actions_one_hot = tf.one_hot(actions, depth=5)
            # policy_loss = self.huber_loss(actions_one_hot, logits, from_logits=True)
            # value_loss = self.huber_loss(discounted_rewards, values)

            actions_one_hot = tf.one_hot(actions, depth=len(logits[0]))
            policy_loss = self.loss_fn(actions_one_hot, logits)

            discounted_rewards = tf.reshape(discounted_rewards, values.shape)

            value_loss = self.loss_fn(discounted_rewards, values)

            total_loss = policy_loss + value_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



class AtomEnv(gym.Env):
    def __init__(self):
        super(AtomEnv, self).__init__()
        rospy.init_node('atom_a2c', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/atom/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/atom/odom', Odometry, self.pose_callback)
        # self.reset_proxy = rospy.ServiceProxy('/reset', Empty)

        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.model_state_msg = ModelState()
        self.model_state_msg.model_name = 'atom_bot'

        self.state = None
        
        self.target_x = 4
        self.target_y = 4

        self.count = 0
        self.agent = TurtlesimActorCriticAgent(num_actions=5)

        self.model_state_msg.pose.position.x = 0.0
        self.model_state_msg.pose.position.y = 0.0
        
        
        self.rate = rospy.Rate(10)  # 10hz


    def pose_callback(self, data):
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        _, _, current_theta = tf_trans.euler_from_quaternion(quaternion)
        
        self.state = [
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            current_theta,
            data.twist.twist.linear.x,
            data.twist.twist.angular.z,
        ]

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_target_position(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def reset(self):
        # self.reset_proxy()
        self.model_state_msg.pose.position.x = 0.0
        self.model_state_msg.pose.position.y = 0.0
        self.model_state_msg.pose.orientation.z = 0.0
        self.model_state_msg.pose.orientation.w = 0.0
        self.set_model_state_proxy(self.model_state_msg)
        self.state = None
        self.count = 0
        self.rate.sleep()
        return np.array(self.state)

    def train_agent(self, num_episodes):
        for episode in range(num_episodes):
            # state = self.reset()
            # self.turtlesim()

            # self.reset_turtlesim()
            # print("reset happened !!!!!")
            # self.set_target_position(np.random.uniform(0, 10), np.random.uniform(0, 10))
            self.set_target_position(4, 4)
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_discounted_rewards = []


            while not rospy.is_shutdown():
                print("episode: ", episode + 1)
                print("present state is: ", self.state)

                if self.state is not None:
                    state = self.state
                    print("present state is: ", state)
                    state = np.array(state)
                    state = tf.convert_to_tensor([state], dtype=tf.float32)
                    print("shape of present state: ", state.shape)

                    current_x, current_y, current_theta, _, _ = self.state
                    print("x: ", current_x)
                    print("y: ", current_y)
                    print("target_x", self.target_x)
                    print("target_y", self.target_y)
                    distance_to_target = self.euclidean_distance(
                        current_x, current_y, self.target_x, self.target_y
                    )
                    print("distance to target: ", distance_to_target)

                    distance_to_bound = self.euclidean_distance(
                        current_x, current_y, 5.544445, 5.544445
                    )
                    print("distance_to_bound: ", distance_to_bound)
                    if (
                        distance_to_bound > 2.5
                    ):  # more than 2.5 radius from start position
                        break
                    if distance_to_target < 0.5:  # Reached target
                        break

                    target_angle = math.atan2(
                        self.target_y - current_y, self.target_x - current_x
                    )
                    print("target angle b4: ", target_angle)
                    if target_angle < 0:
                        target_angle += 2 * math.pi

                    print("target angle: ", target_angle)

                    current_theta = (
                        current_theta
                        if current_theta >= 0
                        else 2 * math.pi + current_theta
                    )
                    print("current theta ", current_theta)
                    # exit()

                    # Calculate relative angle
                    relative_angle = target_angle - current_theta
                    if relative_angle > math.pi:
                        relative_angle -= 2 * math.pi
                    elif relative_angle < -math.pi:
                        relative_angle += 2 * math.pi

                    print("relative angle ", relative_angle)
                    # exit()

                    # Compute action
                    state = np.array(
                        [
                            current_x,
                            current_y,
                            current_theta,
                            distance_to_target,
                            relative_angle,
                        ]
                    )
                    action = self.agent.get_action(state)

                    # Move turtle
                    vel_msg = Twist()
                    vel_msg.linear.x = 1.0  # Constant linear velocity
                    vel_msg.angular.z = action / 3.0  # Scale the action for angular velocity
                    self.velocity_publisher.publish(vel_msg)

                    next_state = self.state
                    reward = -distance_to_target

                    episode_reward += reward
                    episode_states.append(
                        state
                    )  # episode_states contains present state
                    episode_actions.append(action)
                    episode_discounted_rewards.append(reward)

                self.rate.sleep()
            # print(episode_states)
            print("i came out")
            # Stop the turtle
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)

            print("stopping done")

            self.reset()
            self.count+=1
            print("reset happened ",self.count)

            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(episode_discounted_rewards):
                cumulative_reward = reward + 0.9 * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)

            # Normalize discounted rewards
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                np.std(discounted_rewards) + 1e-8
            )

            print("discounted reward: ", discounted_rewards)

            # Convert lists to numpy arrays
            episode_states = np.array(episode_states)
            episode_actions = np.array(episode_actions)

            print("episode_states shape: ", episode_states.shape)

            # Train agent
            self.agent.train(episode_states, episode_actions, discounted_rewards)

            print(f"Episode {episode + 1}: Reward = {episode_reward}")


if __name__ == "__main__":
    controller = AtomEnv()
    controller.train_agent(num_episodes=100)


