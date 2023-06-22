#!/usr/bin/env python3

# pos_x and pos_y seem to be changing 

import gym
from gym import spaces
import rospy
from turtlesim.msg import Pose
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from math import pow, atan2, sqrt
import numpy as np

class TurtleEnv(gym.Env):
    def __init__(self):
        super(TurtleEnv, self).__init__()

        rospy.init_node('turtle_move_rl', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.reset_proxy = rospy.ServiceProxy('/reset', Empty)

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,))

        self.rate = rospy.Rate(2)

        self.goal_x = 3   # destination coordinates
        self.goal_y = 3
        self.position_x = 5.544445  # init coordinates
        self.position_y = 5.544445

        self.episodes = 5
        self.learning_rate = 0.5
        self.discount_factor = 0.99
        self.epsilon = 0.2

        self.q_table = np.zeros((10, 10, 4))

    def reset(self):
        self.reset_proxy()  # Reset the turtlesim simulation
        # self.position_x = 0
        # self.position_y = 0    # reset to init coordinates
        return self.get_observation()

    def step(self, action):
        self.move_turtle(action)
        self.rate.sleep()
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        return obs, reward, done, {}

    def move_turtle(self, action):
        if action == 0:  # Up
            self.move(2.0, 0.0)
        elif action == 1:  # Down
            self.move(-2.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 2.0)
        elif action == 3:  # Right
            self.move(0.0, -2.0)

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

    def get_reward(self):
        distance_to_goal = sqrt(pow(self.goal_x - self.position_x, 2) + pow(self.goal_y - self.position_y, 2))
        reward = -distance_to_goal
        return reward

    def is_done(self):
        distance_threshold = 0.5
        distance_to_goal = sqrt(pow(self.goal_x - self.position_x, 2) + pow(self.goal_y - self.position_y, 2))
        distance_to_bound = sqrt(pow(5.544445 - self.position_x, 2) + pow(5.544445 - self.position_y, 2))
        if distance_to_bound >= 4.5:
            self.reset()
        print("x and y ",self.position_x,self.position_y)
        print("distance_tobound ",distance_to_bound)

        return distance_to_goal < distance_threshold

    def q_learning(self):
        for episode in range(self.episodes):
            state = self.reset()
            done = False

            while not done:
                print("Episode no. : ",episode)
                state_index = self.get_state_index(state)
                print("start state ",state_index)
                print("pos_x, pos_y ",self.position_x,self.position_y)
                action = self.get_action(state_index)
                print("action ",action)

                next_state, reward, done, _ = self.step(action)
                next_state_index = self.get_state_index(next_state)
                print("next start state ",next_state_index)
                print("reward ", reward)

                current_q_value = self.q_table[state_index][action]
                print("old_q_value ",current_q_value)
                max_q_value = np.max(self.q_table[next_state_index])
                print("max q value of present state ", max_q_value)
                new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_q_value)
                print("new_q_value of present state ",new_q_value)

                self.q_table[state_index][action] = new_q_value

                state = next_state

            print(f"Episode: {episode + 1} completed.")

    def get_state_index(self, state):   #current state
        x = int(state[2])  # position_x
        y = int(state[3])  # position_y
        return x, y

    def get_action(self, state_index):
        x, y = state_index
        # return np.argmax(self.q_table[x][y])
        if np.random.random() < self.epsilon:
            return np.argmax(self.q_table[x,y])
        else:
            return np.random.randint(4)

if __name__ == '__main__':
    env = TurtleEnv()
    env.q_learning()
