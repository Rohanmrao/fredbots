import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Actor-Critic model
class ActorCriticModel(keras.Model):
    def __init__(self, num_actions):
        super(ActorCriticModel, self).__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.policy_logits = layers.Dense(num_actions)
        self.dense2 = layers.Dense(32, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):  # call is like a forward function 
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v = self.dense2(inputs)
        values = self.values(v)
        return logits, values


# # Custom Actor-Critic agent for turtlesim
# class TurtlesimActorCriticAgent:
#     def __init__(self, num_actions):
#         self.model = ActorCriticModel(num_actions)
#         self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
#         self.huber_loss = keras.losses.Huber()

#     def get_action(self, state):
#         state = tf.convert_to_tensor([state], dtype=tf.float32)
#         logits, _ = self.model(state)
#         action_probabilities = tf.nn.softmax(logits)
#         action = np.random.choice(len(action_probabilities[0]), p=action_probabilities[0])
#         return action

#     def train(self, states, actions, discounted_rewards):
#         with tf.GradientTape() as tape:
#             logits, values = self.model(states)
#             advantage = discounted_rewards - values

#             actions_one_hot = tf.one_hot(actions, depth=len(self.model.output[0]))
#             policy_loss = self.huber_loss(actions_one_hot, logits, from_logits=True)
#             value_loss = self.huber_loss(discounted_rewards, values)

#             total_loss = policy_loss + value_loss

#         grads = tape.gradient(total_loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


# Custom Actor-Critic agent for turtlesim
class TurtlesimActorCriticAgent:
    def __init__(self, num_actions):
        self.model = ActorCriticModel(num_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        # Huber loss is a type of loss function commonly used in regression tasks, including reinforcement learning.

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.model(state)
        action_probabilities = tf.nn.softmax(logits)
        print("action_probs b4 converting to numpy: ",action_probabilities)
        action_probabilities = action_probabilities[0].numpy()  # Convert to NumPy array
        print("action probability after converting to numpy ", action_probabilities)
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        print("chosen action ", action )
        return action

    def train(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:  # tf.GradientTape() is a TensorFlow API that enables automatic differentiation. 
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
        # self.target_x = None
        # self.target_y = None

        self.target_x = 4
        self.target_y = 4

        self.agent = TurtlesimActorCriticAgent(num_actions=5)
        # actions:
            # Move forward with a moderate linear velocity.
            # Move backward with a moderate linear velocity.
            # Rotate clockwise with a moderate angular velocity.
            # Rotate counterclockwise with a moderate angular velocity.
            # Stop, maintaining the current linear and angular velocities.

        rospy.init_node('turtlebot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        self.state = [data.x, data.y, data.theta, data.linear_velocity, data.angular_velocity]

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_target_position(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def move_turtle(self):
        while not rospy.is_shutdown():
            if self.state is not None:
                current_x, current_y, current_theta, _, _ = self.state
                distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)
                if distance_to_target < 0.5:  # Reached target
                    break

                target_angle = math.atan2(self.target_y - current_y, self.target_x - current_x)
                if target_angle < 0:
                    target_angle += 2 * math.pi

                current_theta = current_theta if current_theta >= 0 else 2 * math.pi + current_theta

                # Calculate relative angle
                relative_angle = target_angle - current_theta
                if relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                elif relative_angle < -math.pi:
                    relative_angle += 2 * math.pi

                # Compute action
                state = np.array([current_x, current_y, current_theta, distance_to_target, relative_angle])
                print("curent state: ", state)
                action = self.agent.get_action(state)

                # Move turtle
                vel_msg = Twist()
                vel_msg.linear.x = 1.0  # Constant linear velocity
                vel_msg.angular.z = action / 2.0  # Scale the action for angular velocity
                self.velocity_publisher.publish(vel_msg)

            self.rate.sleep()

        # Stop the turtle
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(vel_msg)

    def train_agent(self, num_episodes):
        for episode in range(num_episodes):
            self.set_target_position(np.random.uniform(0, 10), np.random.uniform(0, 10))
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_discounted_rewards = []

            while not rospy.is_shutdown():
                print("episode: ",episode+1)
                if self.state is not None:
                    current_x, current_y, current_theta, _, _ = self.state
                    distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)
                    if distance_to_target < 0.5:  # Reached target
                        break

                    target_angle = math.atan2(self.target_y - current_y, self.target_x - current_x)
                    if target_angle < 0:
                        target_angle += 2 * math.pi

                    current_theta = current_theta if current_theta >= 0 else 2 * math.pi + current_theta

                    # Calculate relative angle
                    relative_angle = target_angle - current_theta
                    if relative_angle > math.pi:
                        relative_angle -= 2 * math.pi
                    elif relative_angle < -math.pi:
                        relative_angle += 2 * math.pi

                    # Compute action
                    state = np.array([current_x, current_y, current_theta, distance_to_target, relative_angle])
                    action = self.agent.get_action(state)

                    # Move turtle
                    vel_msg = Twist()
                    vel_msg.linear.x = 1.0  # Constant linear velocity
                    vel_msg.angular.z = action / 2.0  # Scale the action for angular velocity
                    self.velocity_publisher.publish(vel_msg)

                    next_state = self.state
                    reward = -distance_to_target

                    episode_reward += reward
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_discounted_rewards.append(reward)

                self.rate.sleep()

            # Stop the turtle
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)

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
if __name__ == '__main__':
    controller = TurtleBot3Controller()
    controller.train_agent(num_episodes=100)