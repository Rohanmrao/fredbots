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

from tf.transformations import euler_from_quaternion

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

from collections import deque
import random

from fredbots.srv import AddTwoInts
from fredbots.srv import AddTwoIntsRequest




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

    def predict_random(self, inputt, action):
        l = self.model.predict(inputt, verbose=0)[0]
        act= random.choice(l)

        while act == action :
            act= random.choice(l)

        return list(l).index(act)


# Create a class for the environment
class Env():
    def __init__(self, grid_size=6, max_steps=500):

        rospy.init_node("atom_dqn_1", anonymous=True)
        self.velocity_publisher = rospy.Publisher("/atom_2/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber( "/atom_2/odom", Odometry, self.pose_callback)

        self.grid_size = grid_size
        self.max_steps = max_steps
        # self.goal = np.random.randint(0, grid_size, size=2) # random goal
        # print('Goal:', self.goal)

        self.rewards = np.zeros((grid_size, grid_size))
        # self.reset()

        self.steps = 0
        self.pos = [0,0]
        self.done = False
        self.x1 = 0
        self.y1 = 0
        self.theta1 = 0

        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        self.state = [
            data.pose.pose.position.x,
            data.pose.pose.position.y,
        ]
        self.x1 = data.pose.pose.position.x
        self.y1 = data.pose.pose.position.y
        roll, pitch, self.theta1 = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

        # self.theta1 is the yaw 

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
    

    def step(self, action, state): # As per the paper [REWARD FUNCTION]
        self.pos = state 
        self.steps += 1
        prev_pos = self.pos.copy()
        if action == 0 and self.pos[0] < self.grid_size - 1: # right
            self.pos[0] += 1
        elif action == 1 and self.pos[0] > 0: # left
            self.pos[0] -= 1
        elif action == 2 and self.pos[1] > 0: # down
            self.pos[1] -= 1
        elif action == 3 and self.pos[1] < self.grid_size - 1: # up
            self.pos[1] += 1
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
    


    def rotate (self, angular_speed_degree, relative_angle_degree, clockwise, goal_angle):
        
        velocity_message = Twist()
        velocity_message.linear.x=0
        velocity_message.linear.y=0
        velocity_message.linear.z=0
        velocity_message.angular.x=0
        velocity_message.angular.y=0
        velocity_message.angular.z=0

        angular_speed=math.radians(abs(angular_speed_degree))

        if (clockwise):
            velocity_message.angular.z =-abs(angular_speed)
        else:
            velocity_message.angular.z =abs(angular_speed)

        angle_moved = 0.0
        loop_rate = rospy.Rate(10) # we publish the velocity at 10 Hz (10 times a second)    
        # cmd_vel_topic='/cmd_vel_mux/input/teleop'
        # pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        t0 = rospy.Time.now().to_sec()

        while True :
            # rospy.loginfo("Turtlesim rotates")
            self.velocity_publisher.publish(velocity_message)

            t1 = rospy.Time.now().to_sec()
            current_angle_degree = (t1-t0)*angular_speed_degree
            loop_rate.sleep()

            # print 'current_angle_degree: ',current_angle_degree
            if self.angle_difference(math.degrees(self.theta1), goal_angle) < 1:
                rospy.loginfo("Atom_2 rotated")
                break

            # if  (current_angle_degree>relative_angle_degree):
            #     rospy.loginfo("Atom_2 rotated")
            #     break

        #finally, stop the robot when the distance is moved
        velocity_message.angular.z =0
        self.velocity_publisher.publish(velocity_message)

    def angle_difference(self, angle1, angle2):
    
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    
    def Goto_goal(self, x_goal, y_goal):
        
        print("NOW ILL MOVE")
        msg = Twist()

        distance = abs(math.sqrt(((x_goal-self.x1)**2)+((y_goal-self.y1)**2)))

        angle_t = math.atan2((y_goal-self.y1),(x_goal-self.x1)) # -pi to pi
        angle_t = math.degrees(angle_t) # -180 to 180


        theta1_deg = math.degrees(self.theta1) # -180 to 180
        # angle = offset between current and goal angle ranging from -180 to 180
        if angle_t < 0:
            angle_t_b = 360 + angle_t # 180 to 360
        else:
            angle_t_b = angle_t # 0 to 180
        if theta1_deg < 0:
            theta1_deg_b = 360 + theta1_deg # 180 to 360
        else:
            theta1_deg_b = theta1_deg # 0 to 180

        angle = angle_t_b - theta1_deg_b # -360 to 360
        


        print(theta1_deg, angle_t, angle)
        phi = 1
        if -180 < angle < 0:
            self.rotate(min(abs(angle*phi),30), abs(angle), True, angle_t)
        elif -360 < angle < -180:
            angle = 360 - abs(angle)
            self.rotate(min(abs(angle*phi),30), abs(angle), False, angle_t)
        elif 180 < angle < 360:
            angle = 360 - angle
            self.rotate(min(abs(angle*phi),30), angle, True, angle_t) 
        else:
            self.rotate(min(abs(angle*phi),30), angle, False, angle_t)

        while (distance>0.15):        
            Beta = 0.5
            distance = abs(math.sqrt(((x_goal-self.x1)**2)+((y_goal-self.y1)**2)))

            speed = distance*Beta

            msg.linear.x = speed
            msg.angular.z = 0

            self.velocity_publisher.publish(msg)

        msg.linear.x = 0
        msg.angular.z = 0
        self.velocity_publisher.publish(msg)
            

    def euclidean_distance_from_goal(self, pos):
        dist = np.sqrt(np.sum((pos - self.goal) ** 2))
        return dist


            


if __name__ == "__main__":
    model = tf.keras.models.load_model('/home/amulya/catkin_ws/src/fredbots/src/scripts/new/frozen_lake/diff_target/doublenet_dqn_diff_start_diff_goal_300ep_pt2.h5')
    print("Loaded model from disk")
    agent = Agent(Env(), model=model, target_model=model)

    # testing for custom goal and random start point:
    # state = agent.env.reset()
    state = [0,5]
    goal = agent.env.reset_goal()
    print('Goal:', goal)

    flag = True

    for step in range(30):
        print('State:', state)
        inputt = np.concatenate((state, goal))
        inputt = tf.convert_to_tensor(inputt)
        inputt = tf.expand_dims(inputt, 0)

        x = state.copy()

        if flag:
            action = agent.predict(inputt)
            print("action: ", action)
            next_state, reward, done, terminate = agent.env.step(action, state)
            print("next_state after action: ", next_state)
        
        state = x.copy()
        # [LOCAL CONTROLLER]
        rospy.wait_for_service('add_two_ints')
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)

        request = AddTwoIntsRequest()
        request.cur_x = state[0]
        request.cur_y = state[1]

        request.next_x = next_state[0]
        request.next_y = next_state[1]

        response = add_two_ints(request)

        print("state: ", state)
        print("NEXT_STATE: ", next_state)

        if (response.occ == 0): # [NOT OCCUPIED]
            flag = True
            print("occu: ", response.occ)
            agent.env.Goto_goal(next_state[0], next_state[1])
        
        else: # [IF OCCUPIED]
            agent.env.pos = state.copy()
            print("occu: ", response.occ)
            flag = False
            agent.predict_random(inputt, action)
            print("action after random prediction: ", action)
            next_state, reward, done, terminate = agent.env.step(action, state)
            print("next_state after random action: ", next_state)
            print("\n\n")
            continue

        state = next_state
        print("next state: ", next_state)
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

        print("\n*****************\n")


