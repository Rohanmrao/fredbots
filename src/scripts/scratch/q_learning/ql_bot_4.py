#!/usr/bin/env python3

import math
import rospy
import numpy as np
import time

from turtlesim.srv import TeleportAbsolute
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion

env_row = 11
env_col = 11

# define actions
# numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

q_values = np.zeros((env_row, env_col, len(actions)))

# initialise rewards
rewards = np.full((env_row, env_col), -100.)

goal_x = 6  # int(input("Please enter goal x coordinate: "))
goal_y = 3  # int(input("Please enter goal y coordinate: "))

obstacles = []

#define training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the agent should learn

x1 = 0
y1 = 0
z1 = 0
theta1 = 0


def get_euclidian_dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

# assign reward value to goal
for i in range(len(q_values)):
    for j in range(len(q_values[0])):
        rewards[i][j] = - get_euclidian_dist(i,j,goal_x,goal_y)

# Functions for QLearning   

def is_final_state(row, col):
    if row == goal_x and col == goal_y:
        return True
    else:
        return False
    
#define a function that will choose a random, non-terminal starting location
def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(env_row)
  current_column_index = np.random.randint(env_col)

  return current_row_index, current_column_index

#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon, 
  #then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(4)
  
#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < env_col - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < env_row - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

def reset_atom():
    set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_msg = ModelState()
    set_model_state_msg.model_name = 'atom_4'
    set_model_state_msg.pose.position.x = start_pos_x
    set_model_state_msg.pose.position.y = start_pos_y
    set_model_state_msg.pose.position.z = 0
    set_model_state_msg.pose.orientation.x = 0
    set_model_state_msg.pose.orientation.y = 0
    set_model_state_msg.pose.orientation.z = 0
    set_model_state_msg.pose.orientation.w = 1
    set_model_state_proxy(set_model_state_msg)

def stop_atom():
    pub = rospy.Publisher('/atom/cmd_vel', Twist, queue_size=1)
    twist = Twist()
    pub.publish(twist)

#run through 1000 training episodes
def train():
  print("Obstacles are at:" + str(obstacles))
  for obstacle in obstacles:
    rewards[obstacle[0]][obstacle[1]] = -100 #remove the reward for any actions that lead to the obstacle
  for episode in range(1000):
    #get the starting location for this episode
    row_index, column_index = get_starting_location()
    #continue taking actions (i.e., moving) until we reach a terminal state
    #(i.e., until we reach the item packaging area or crash into an item storage location)
    while not is_final_state(row_index, column_index):
      #choose which action to take (i.e., where to move next)
      action_index = get_next_action(row_index, column_index, epsilon)
      #perform the chosen action, and transition to the next state (i.e., move to the next location)
      old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
      row_index, column_index = get_next_location(row_index, column_index, action_index)
      #receive the reward for moving to the new state, and calculate the temporal difference
      reward = rewards[row_index, column_index]
      old_q_value = q_values[old_row_index, old_column_index, action_index]
      temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
      #update the Q-value for the previous state and action pair
      new_q_value = old_q_value + (learning_rate * temporal_difference)
      q_values[old_row_index, old_column_index, action_index] = new_q_value
  print('Training complete!')

#Define a function that will get the shortest path between any location within the warehouse that 
#the robot is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_column_index, atom_sim=False):
  #return immediately if this is an invalid starting location
  train()
  if is_final_state(start_row_index, start_column_index):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_final_state(current_row_index, current_column_index):
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
      global obstacles
      obstacles = []
      if atom_sim:
        print("Moving to location: ", current_row_index, current_column_index)
        Goto_goal(current_row_index, current_column_index)
    
    return shortest_path

# ROS movement functions and callbacks

def laser_callback(msg):

    laser_range = msg.ranges

    coord = update_obst_coord(laser_range, msg.angle_min, msg.angle_max, len(laser_range))

    approx_new_coord(coord)

def pose_callback(data):
    global x1,y1,z1,theta1
    x1 = data.pose.pose.position.x
    y1 = data.pose.pose.position.y
    roll, pitch, theta1 = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

def Goto_goal(x_goal, y_goal):
    global x1,y1,theta1
    msg = Twist()


    rate = rospy.Rate(10)

    distance = abs(math.sqrt(((x_goal-x1)**2)+((y_goal-y1)**2)))

    ang_dist = math.atan2((y_goal-y1),(x_goal-x1))
    while (abs(ang_dist-theta1)>0.05):
        Phi = 4
        ang_dist = math.atan2((y_goal-y1),(x_goal-x1))

        ang_speed = Phi*(ang_dist-theta1)

        msg.linear.x = 0
        msg.angular.z = ang_speed

        pub.publish(msg)

    # rospy.sleep(2)

    while (distance>0.15):        
        Beta = 0.75
        distance = abs(math.sqrt(((x_goal-x1)**2)+((y_goal-y1)**2)))

        speed = distance*Beta

        msg.linear.x = speed
        msg.angular.z = 0

        pub.publish(msg)

    msg.linear.x = 0
    msg.angular.z = 0
    pub.publish(msg)

def update_obst_coord(range_data, theta_min, theta_max, num_samples):

    # Get the angle of each sample
    thetas = np.linspace(theta_min, theta_max, num_samples)

    # Adjust theta according to the orientation of the robot
    thetas = theta1 + thetas

    # Get the apparent x and y coordinates of each sample
    xs, ys = [], []
    for i in range(num_samples):
        if range_data[i] < 5:
            xs.append(range_data[i] * np.cos(thetas[i]))
            ys.append(range_data[i] * np.sin(thetas[i]))
        else:
            xs.append(float('inf'))
            ys.append(float('inf'))

    # Calculate the lidar's offset from the robot's center
    l_x = x1 + 0.2 * np.cos(theta1)
    l_y = y1 + 0.2 * np.sin(theta1)

    # Get the x and y coordinates of each sample
    obst_coord = [(xs[i]+l_x, ys[i]+l_y) for i in range(num_samples)]

def approx_new_coord(coord):
    flag = 0
    if not coord:
        return
    for (i,j) in coord:
        if (i == float('inf')) or (i == float('-inf')) or (j == float('inf')) or (j == float('-inf')):
            continue
        else:
            i_r = int(round(i,0))
            j_r = int(round(j,0))
            if [i_r,j_r] in obstacles:
                continue
            else:
                obstacles.append([i_r,j_r])
                flag = 1
    if flag == 1:
        train()

rospy.init_node("Shortest_path_atom_4")
pub = rospy.Publisher("/atom_4/cmd_vel",Twist, queue_size =10)
sub = rospy.Subscriber("/atom_4/odom", Odometry, pose_callback)
las = rospy.Subscriber("/scan_4", LaserScan, laser_callback)

time.sleep(1)

start_pos_x = int(round(x1,0))
start_pos_y = int(round(y1,0))
print(start_pos_x, start_pos_y)

get_shortest_path(start_pos_x, start_pos_y, atom_sim=True)