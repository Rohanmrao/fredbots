#!/usr/bin/env python3

import gym
from gym import spaces
import rospy
from turtlesim.msg import Pose
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from math import pow, atan2, sqrt

class TurtleEnv(gym.Env):
    def __init__(self):
        super(TurtleEnv, self).__init__()

        rospy.init_node('turtle_move_rl', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.reset_proxy = rospy.ServiceProxy('/reset', Empty)

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,))

        self.rate = rospy.Rate(10)

        self.goal_x = 10    # destination coordinates
        self.goal_y = 10
        self.position_x = 0  # init coordinates 
        self.position_y = 0

    def reset(self):
        self.reset_proxy()  # Reset the turtlesim simulation
        # self.goal_x = rospy.get_param('~goal_x')
        # self.goal_y = rospy.get_param('~goal_y')
        self.position_x = 0
        self.position_y = 0    #reset to init coordinates
        return self.get_observation()

    def step(self, action):
        if action == 0:  # Up
            self.move(1.0, 0.0)
        elif action == 1:  # Down
            self.move(-1.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 1.0)
        elif action == 3:  # Right
            self.move(0.0, -1.0)

        self.rate.sleep()
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        return obs, reward, done, {}

    def move(self, linear_vel, angular_vel):
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_vel
        velocity_msg.angular.z = angular_vel
        self.velocity_publisher.publish(velocity_msg)

    def update_pose(self, data):
        self.position_x = round(data.x, 2)
        self.position_y = round(data.y, 2)

    def get_observation(self):
        return [self.goal_x, self.goal_y, self.position_x, self.position_y]

    def get_reward(self): # use trial and error to minimise the distance between the goal and current pose using euclidian distance
        distance_to_goal = sqrt(pow(self.goal_x - self.position_x, 2) + pow(self.goal_y - self.position_y, 2))
        reward = -distance_to_goal
        return reward

    def is_done(self):
        distance_threshold = 0.1
        distance_to_goal = sqrt(pow(self.goal_x - self.position_x, 2) + pow(self.goal_y - self.position_y, 2))
        return distance_to_goal < distance_threshold


if __name__ == '__main__':

    # rospy.init_node('turtle_move_rl')

    env = TurtleEnv()

    episodes = 100

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample() #pick a random action 
            # print(f"Episode: {episode + 1}, Reward: {total_reward}")
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                print(f"Episode: {episode + 1}, Reward: {total_reward}")
