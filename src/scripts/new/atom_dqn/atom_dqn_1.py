#!/usr/bin/env python3
import math
import os
import numpy as np
import rospy
import tensorflow as tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tensorflow import keras
from tensorflow.keras import layers
import tf.transformations as tf_trans

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

from collections import deque
import random




# Create an agent class
class Agent():
    def __init__(self, env, model, target_model):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.target_model.set_weights(self.model.get_weights())
        self.gamma = 0.7
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.01
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        

    def predict(self, inputt):

        return np.argmax(self.model.predict(inputt, verbose=0)[0])



# Create a class for the environment
class Env():
    def __init__(self, grid_size=6, max_steps=500):

        rospy.init_node("atom_dqn", anonymous=True)
        self.velocity_publisher = rospy.Publisher("/atom/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber( "/atom/odom", Odometry, self.pose_callback)

        self.grid_size = grid_size
        self.max_steps = max_steps
        # self.goal = np.random.randint(0, grid_size, size=2) # random goal
        # print('Goal:', self.goal)

        self.rewards = np.zeros((grid_size, grid_size))
        # self.reset()

        self.steps = 0
        self.pos = [0,0]
        self.done = False

        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        global x,y
        self.state = [
            data.pose.pose.position.x,
            data.pose.pose.position.y,
        ]
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

    def reset(self):
        self.pos = np.random.randint(0, self.grid_size, size=2)
        self.steps = 0
        self.done = False
        return self.pos
    
    def reset_goal(self):
        self.goal = np.random.randint(0, self.grid_size, size=2)
        # print('Goal:', self.goal)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.rewards[i, j] = -self.euclidean_distance_from_goal(np.array([i, j]))
        self.rewards[self.goal[0], self.goal[1]] = 100
        return self.goal
    

    def step(self, action): # As per the paper
        self.steps += 1
        prev_pos = self.pos.copy()
        if action == 0 and self.pos[0] < self.grid_size - 1: # right
            self.pos[0] += 1
            self.move(0.0, -3.0, self.pos[0], self.pos[1])
        elif action == 1 and self.pos[0] > 0: # left
            self.pos[0] -= 1
            self.move(0.0, 3.0, self.pos[0], self.pos[1])
        elif action == 2 and self.pos[1] > 0: # down
            self.pos[1] -= 1
            self.move(-3.0, 0.0, self.pos[0], self.pos[1])
        elif action == 3 and self.pos[1] < self.grid_size - 1: # up
            self.pos[1] += 1
            self.move(3.0, 0.0, self.pos[0], self.pos[1])
        else:
            reward = -150
            self.done = True
            return self.pos, reward, self.done, True # TODO: The episode is not terminated.
        if np.array_equal(self.pos, self.goal):
            self.done = True
            reward = 500
        elif self.steps >= self.max_steps:
            self.done = True
            reward = self.rewards[self.pos[0], self.pos[1]]
        else:
            if self.euclidean_distance_from_goal(self.pos) < self.euclidean_distance_from_goal(prev_pos):
                reward = 10
            else:
                reward = -10
        return self.pos, reward, self.done, False
    
    
    def move(self, linear_vel, angular_vel, pos_x, pos_y):
        global x,y
        print("printing pos_x and pos_y",pos_x, pos_y)
        # print("x,y: ",x,y)
        count = 0
        while(True):
            print("x,y: ",x,y)
            velocity_msg = Twist()
            velocity_msg.linear.x = linear_vel
            velocity_msg.angular.z = angular_vel
            self.velocity_publisher.publish(velocity_msg)
            
            distance = abs(math.sqrt(((pos_x-x)**2)+((pos_y-y)**2)))

            # print("distance: ", distance)
            # count+=1
            if (distance<0.1):
                break
        

    def euclidean_distance_from_goal(self, pos):
        dist = np.sqrt(np.sum((pos - self.goal) ** 2))
        return dist


            


if __name__ == "__main__":
    model = tf.keras.models.load_model('/home/amulya/catkin_ws/src/fredbots/src/scripts/new/frozen_lake/diff_target/doublenet_dqn_diff_start_diff_goal_300ep_pt2.h5')
    print("Loaded model from disk")
    agent = Agent(Env(), model=model, target_model=model)

    # testing for custom goal and random start point:
    # state = agent.env.reset()
    state = [0,0]
    goal = agent.env.reset_goal()
    print('Goal:', goal)

    for step in range(30):
        print('State:', state)
        inputt = np.concatenate((state, goal))
        inputt = tf.convert_to_tensor(inputt)
        inputt = tf.expand_dims(inputt, 0)
        action = agent.predict(inputt)
        next_state, reward, done, terminate = agent.env.step(action)
        state = next_state
        if done:
            print('State:', state)
            print('Steps: ', step+1)
            if np.array_equal(agent.env.goal, agent.env.pos):
                print('Reached the goal!')
            break
        if terminate:
            print('Steps: ', step+1)
            print("Crashed into a wall")
            break


