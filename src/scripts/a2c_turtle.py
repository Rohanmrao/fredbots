import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np


class TurtleBot3Controller:
    def __init__(self):
        rospy.init_node('turtlebot3_controller', anonymous=True)
        self.state = None
        self.target_x = 0
        self.target_y = 0
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/pose', Pose, self.pose_callback)
        self.reset_proxy = rospy.ServiceProxy('/reset', Empty)
        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        # Callback function for pose subscriber, updates the current state of the turtle
        self.state = [data.x, data.y, data.theta, data.linear_velocity, data.angular_velocity]

    def euclidean_distance(self, x1, y1, x2, y2):
        # Calculates the Euclidean distance between two points
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_target_position(self, target_x, target_y):
        # Sets the target position for the turtle
        self.target_x = target_x
        self.target_y = target_y

    def reset_turtlesim(self):
        # Calls the reset service to reset the turtlesim environment
        rospy.wait_for_service('/reset')
        try:
            reset_service = rospy.ServiceProxy('/reset', Empty)
            reset_service()
            rospy.sleep(1.0)
        except rospy.ServiceException as e:
            print("Reset service call failed:", str(e))

    def move_turtle(self, action):
        # Moves the turtle based on the given action
        if action == 0:  # Up
            self.move(3.0, 0.0)
        elif action == 1:  # Down
            self.move(-3.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 3.0)
        elif action == 3:  # Right
            self.move(0.0, -3.0)

    def move(self, linear_vel, angular_vel):
        # Publishes the linear and angular velocities to move the turtle
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_vel
        velocity_msg.angular.z = angular_vel
        self.velocity_publisher.publish(velocity_msg)

    def train_agent(self, num_episodes):
        for episode in range(num_episodes):
            self.set_target_position(4, 4)
            episode_reward = 0
            episode_states = []
            episode_actions = []
            episode_discounted_rewards = []

            while not rospy.is_shutdown():
                if self.state is not None:
                    state = self.state

                    current_x, current_y, current_theta, _, _ = self.state
                    distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)

                    distance_to_bound = self.euclidean_distance(current_x, current_y, 5.544445, 5.544445)
                    
                    if distance_to_bound > 3.5:  # more than 2.5 radius from the start position
                        reward = -distance_to_target
                        break
                    if distance_to_target < 0.5:  # Reached the target
                        reward = -distance_to_target + 100
                        break

                    target_angle = math.atan2(self.target_y - current_y, self.target_x - current_x)
                    theta_diff = target_angle - current_theta

                    if theta_diff > math.pi:
                        theta_diff -= 2 * math.pi
                    elif theta_diff < -math.pi:
                        theta_diff += 2 * math.pi

                    action = self.agent.get_action(state)
                    self.move_turtle(action)

                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_reward += reward

                    self.rate.sleep()

            # Stop the turtle
            vel_msg = Twist()
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.velocity_publisher.publish(vel_msg)

            self.reset_turtlesim()

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

        self.agent.model.save_weights('model/model.h5')


# Example usage
if __name__ == '__main__':
    controller = TurtleBot3Controller()
    controller.train_agent(num_episodes=100)
