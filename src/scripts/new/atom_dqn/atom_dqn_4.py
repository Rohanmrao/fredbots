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
        self.velocity_publisher = rospy.Publisher("/atom_4/cmd_vel", Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber( "/atom_4/odom", Odometry, self.pose_callback)

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
            # self.move(0.0, -3.0, self.pos[0], self.pos[1])
        elif action == 1 and self.pos[0] > 0: # left
            self.pos[0] -= 1
            # self.move(0.0, 3.0, self.pos[0], self.pos[1])
        elif action == 2 and self.pos[1] > 0: # down
            self.pos[1] -= 1
            # self.move(-3.0, 0.0, self.pos[0], self.pos[1])
        elif action == 3 and self.pos[1] < self.grid_size - 1: # up
            self.pos[1] += 1
        else:
            reward = -150
            self.done = True
            return self.pos, reward, self.done, True # TODO: The episode is not terminated.
        self.Goto_goal(self.pos[0], self.pos[1])
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
    
    # def update_pose(self):
    #     global x, y
    #     x = self.state[0]
    #     y = self.state[1]
    
    # def move(self, linear_vel, angular_vel, pos_x, pos_y):
    #     global x,y
    #     print("printing pos_x and pos_y",pos_x, pos_y)
    #     # print("x,y: ",x,y)
    #     count = 0
    #     while(True):
    #         print("x,y: ",x,y)
    #         velocity_msg = Twist()
    #         velocity_msg.linear.x = linear_vel
    #         velocity_msg.angular.z = angular_vel
    #         self.velocity_publisher.publish(velocity_msg)

    #         # self.update_pose()
    #         print("\n\n", x, y, "\n\n")
            
    #         distance = abs(math.sqrt(((pos_x-x)**2)+((pos_y-y)**2)))

    #         # print("distance: ", distance)
    #         count+=1
    #         if (distance<0.1):
    #             break

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
        
        
        msg = Twist()

        distance = abs(math.sqrt(((x_goal-self.x1)**2)+((y_goal-self.y1)**2)))

        # ang_dist = math.atan2((y_goal-y1),(x_goal-x1))
        # while (abs(ang_dist-theta1)>0.01):
        #     Phi = 4
        #     ang_dist = math.atan2((y_goal-y1),(x_goal-x1))

        #     ang_speed = Phi*(ang_dist-theta1)

        #     msg.linear.x = 0
        #     msg.angular.z = ang_speed

        #     pub.publish(msg)

        angle_t = math.atan2((y_goal-self.y1),(x_goal-self.x1)) # -pi to pi
        angle_t = math.degrees(angle_t) # -180 to 180

        # if angle_t < 0:
        #     angle_t = 360 + angle_t

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
        
        # theta1_deg = math.degrees(theta1)
        # angle = angle - theta1_deg
        # rotate(10, angle, True)

        # rospy.sleep(2)

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
    state = [2,2]
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

