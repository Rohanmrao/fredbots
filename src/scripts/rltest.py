# test out a basic RL code snippet here

import gym
from gym import spaces
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import module_env

# Initialize the ROS node
rospy.init_node('turtlesim_rl')

# Define the action and observation spaces
action_space = spaces.Discrete(2)  # Turn Left, Turn Right
observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=float)  # X and Y position

# Initialize the current position
current_pose = None


def pose_callback(msg):
    global current_pose
    current_pose = (msg.x, msg.y)

def take_action(action):
    vel_cmd = Twist()
    if action == 0:  # Turn Left
        vel_cmd.angular.z = 1.0
    elif action == 1:  # Turn Right
        vel_cmd.angular.z = -1.0
    vel_pub.publish(vel_cmd)

def calculate_reward():
    # You can define your own reward function based on the desired behavior
    return 1.0

def get_observation():
    return current_pose


# Set up ROS publisher and subscriber
vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
pose_sub = rospy.Subscriber('/turtle1/pose', Pose, pose_callback)

if __name__ == '__main__':
    env = gym.make('TurtlesimRL-v0')
    env.action_space = action_space
    env.observation_space = observation_space

    done = False
    observation = env.reset()

    while not done:
        action = env.action_space.sample()  # Replace with your RL algorithm's action selection
        observation, reward, done, _ = env.step(action)
