import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import TeleportAbsolute
from turtlesim.srv import SetPen
from std_srvs.srv import Empty
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# changing actions to just FRONT, BACK, LEFT, RIGHT


# Actor model
class Actor(keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.dense3 = layers.Dense(32, activation="relu")
        self.policy_logits = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        logits = self.policy_logits(x)
        return logits


# Critic model
class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.dense3 = layers.Dense(32, activation="relu")
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
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
        # Huber loss is a type of loss function commonly used in regression tasks, including reinforcement learning.
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        # state is the input
        logits, _ = self.model(state)
        action_probabilities = tf.nn.softmax(logits)
        # Convert to NumPy array
        action_probabilities = action_probabilities[0].numpy()
        action = np.random.choice(
            len(action_probabilities), p=action_probabilities)
        return action

    def train(self, states, actions, discounted_rewards):
        # tf.GradientTape() is a TensorFlow API that enables automatic differentiation.
        with tf.GradientTape() as tape:
            # print("shape of states to train fn: ", states.shape)
            logits, values = self.model(states)
            advantage = discounted_rewards - values
            actions_one_hot = tf.one_hot(actions, depth=len(logits[0]))
            policy_loss = self.loss_fn(actions_one_hot, logits)
            discounted_rewards = tf.reshape(discounted_rewards, values.shape)
            value_loss = self.loss_fn(discounted_rewards, values)
            total_loss = policy_loss + value_loss + 0.01 * advantage

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        self.model.save('model')

# TurtleBot3 Controller


class TurtleBot3Controller:
    def __init__(self):
        self.state = None
        self.max_steps = 200
        self.target_x = 4
        self.target_y = 4
        self.count = 0
        self.agent = TurtlesimActorCriticAgent(num_actions=4)
        rospy.init_node("turtlebot_controller", anonymous=True)
        self.velocity_publisher = rospy.Publisher(
            "/turtle1/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(
            "/turtle1/pose", Pose, self.pose_callback)
        self.reset_proxy = rospy.ServiceProxy("/reset", Empty)
        self.teleport_absolute = rospy.ServiceProxy( "/turtle1/teleport_absolute", TeleportAbsolute)
        self.pen_service = rospy.ServiceProxy("/turtle1/set_pen", SetPen)
        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        self.state = [
            data.x,
            data.y,
            data.theta,
            data.linear_velocity,
            data.angular_velocity,
        ]

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_target_position(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def reset_turtlesim(self):
        rospy.wait_for_service("/reset")
        try:
            reset_service = rospy.ServiceProxy("/reset", Empty)
            reset_service()
            rospy.sleep(1.0)
        except rospy.ServiceException as e:
            print("Reset service call failed:", str(e))

    def teleport_turtle(self, x, y, theta):
        rospy.wait_for_service("/turtle1/teleport_absolute")
        try:
            self.pen_service(off=1)
            self.teleport_absolute(x, y, theta)
            self.pen_service(off=0)
        except rospy.ServiceException as e:
            print("Reset service call failed:", str(e))


    def move_turtle(self, action):
        if action == 0:  # Up
            self.move(3.0, 0.0)
        elif action == 1:  # Down
            self.move(-3.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 3.0)
        elif action == 3:  # Right
            self.move(0.0, -3.0)

    def move(self, linear_vel, angular_vel):
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_vel
        velocity_msg.angular.z = angular_vel
        self.velocity_publisher.publish(velocity_msg)

    def get_random_position(self):
        x = random.uniform(0, 8.5)
        y = random.uniform(0, 10.5)
        while (self.euclidean_distance(x, y, 5.445, 5.445) > 3):
            x = random.uniform(0, 8.5)
            y = random.uniform(0, 10.5)
        return x, y

    def train_agent(self, num_episodes):
        for episode in range(num_episodes):
            self.set_target_position(4, 4)
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_discounted_rewards = []

            start_x, start_y = self.get_random_position()
            self.teleport_turtle(start_x, start_y, 0)

            self.step = 0

            print("episode: ", episode + 1)
            while not rospy.is_shutdown():
                if self.state is not None :
                    self.step+=1
                    print("printing steps: ", self.step)
                    counter = 0                                                                                                    # counter to check if the bot is stuck
                    reward = 0
                    state = self.state
                    state = np.array(state)
                    state = tf.convert_to_tensor([state], dtype=tf.float32)

                    current_x, current_y, current_theta, _, _ = self.state
                    distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)
                    distance_to_bound = self.euclidean_distance(current_x, current_y, 5.544445, 5.544445)

                    if distance_to_bound > 3.5:                                                                                        # more than 2.5 radius from start position
                        reward += -(distance_to_target ** 2)
                        counter = 1
                    if distance_to_target < 0.5:                                                                                        # Reached target
                        reward += -distance_to_target + 100
                        counter = 1

                    target_angle = math.atan2(self.target_y - current_y, self.target_x - current_x)

                    if target_angle < 0:
                        target_angle += 2 * math.pi

                    current_theta = (
                        current_theta
                        if current_theta >= 0
                        else 2 * math.pi + current_theta
                    )

                    relative_angle = target_angle - current_theta
                    if relative_angle > math.pi:
                        relative_angle -= 2 * math.pi
                    elif relative_angle < -math.pi:
                        relative_angle += 2 * math.pi

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

                    # either 0,1,2,3
                    # Move turtle withnew actions
                    self.move_turtle(action)
                    next_state = self.state
                    reward += -distance_to_target
                    try:
                        reward += prev_distance_to_target - distance_to_target
                    except:
                        pass 
                    episode_reward += reward
                    # episode_states contains present state
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_discounted_rewards.append(reward)
                    prev_distance_to_target = distance_to_target

                    if counter == 1:
                        break

                self.rate.sleep()

            # Stop the turtle
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)
            # self.reset_turtlesim()
            self.count += 1

            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(episode_discounted_rewards):
                cumulative_reward = reward + 0.9 * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)

            # Normalize discounted rewards
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

            # Convert lists to numpy arrays
            episode_states = np.array(episode_states)
            episode_actions = np.array(episode_actions)

            # Train agent
            self.agent.train(episode_states, episode_actions, discounted_rewards) 
            print(f"Episode {episode + 1}: Reward = {episode_reward}")


# Example usage
if __name__ == "__main__":
    controller = TurtleBot3Controller()
    controller.train_agent(num_episodes=100)
