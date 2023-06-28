import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Actor model
class Actor(keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
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
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
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
        self.model = ActorCriticModel(num_actions)  # Instantiate the model using the ActorCriticModel class
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.model(state)
        action_probabilities = tf.nn.softmax(logits)
        action_probabilities = action_probabilities[0].numpy()
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        return action

    def train(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            advantage = discounted_rewards - values

            actions_one_hot = tf.one_hot(actions, depth=len(self.model.output[0]))
            policy_loss = self.huber_loss(actions_one_hot, logits, from_logits=True)
            value_loss = self.huber_loss(discounted_rewards, values)

            total_loss = policy_loss + value_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

# TurtleBot3 Controller
class TurtleBot3Controller:
    def __init__(self):
        self.state = None
        self.target_x = 4
        self.target_y = 4

        self.agent = TurtlesimActorCriticAgent(num_actions=5)
        rospy.init_node('turtlebot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        self.state = [data.x, data.y, data.theta, data.linear_velocity, data.angular_velocity]

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_reward(self):
        distance_to_target = self.euclidean_distance(self.state[0], self.state[1], self.target_x, self.target_y)
        if distance_to_target < 0.5:
            reward = 10
        else:
            reward = -1
        return reward

    def run(self):
        total_reward = 0
        while not rospy.is_shutdown():
            action = self.agent.get_action(self.state)
            vel_msg = Twist()
            if action == 0:
                vel_msg.linear.x = 1
                vel_msg.angular.z = 0
            elif action == 1:
                vel_msg.linear.x = -1
                vel_msg.angular.z = 0
            elif action == 2:
                vel_msg.linear.x = 0
                vel_msg.angular.z = 1
            elif action == 3:
                vel_msg.linear.x = 0
                vel_msg.angular.z = -1
            else:
                vel_msg.linear.x = 0
                vel_msg.angular.z = 0

            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()

            reward = self.get_reward()
            total_reward += reward

            next_state = self.state
            if total_reward < 0:
                done = True
            else:
                done = False

            self.agent.train(self.state, action, reward, next_state, done)
            self.state = next_state

            if done:
                rospy.loginfo("Episode finished with a total reward of %i", total_reward)
                break

if __name__ == '__main__':
    controller = TurtleBot3Controller()
    controller.run()
