import gym
from gym import spaces
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

class TurtlesimEnv(gym.Env):
    def __init__(self):
        super(TurtlesimEnv, self).__init__()

        # Initialize the ROS node
        rospy.init_node('turtlesim_gym')

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4)  # Forward, Backward, Turn Left, Turn Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=float)  # X and Y position

        # Set up ROS publisher and subscriber
        self.vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

        # Initialize the current position
        self.current_pose = None
    
    def step(self, action):
        # Take action and execute the corresponding velocity command
        vel_cmd = Twist()
        if action == 0:  # Forward
            vel_cmd.linear.x = 1.0
        elif action == 1:  # Backward
            vel_cmd.linear.x = -1.0
        elif action == 2:  # Turn Left
            vel_cmd.angular.z = 1.0
        elif action == 3:  # Turn Right
            vel_cmd.angular.z = -1.0
        self.vel_pub.publish(vel_cmd)

        # Wait for the next observation
        rospy.sleep(0.1)

        # Calculate the reward (e.g., based on reaching a goal)
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.check_done()

        # Return the observation, reward, and done flag
        observation = self.get_observation()
        return observation, reward, done, {}
    
    def reset(self):
        # Reset the turtle's position
        reset_cmd = Twist()
        reset_cmd.linear.x = -2.0
        self.vel_pub.publish(reset_cmd)
        rospy.sleep(0.1)

        # Return the initial observation
        observation = self.get_observation()
        return observation

# Register the custom environment with Gym
gym.register(
    id='TurtlesimRL-v0',
    entry_point='module_env.py:TurtlesimEnv',
)