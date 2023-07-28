#!/usr/bin/python3

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

from fredbots.srv import AddTwoInts
from fredbots.srv import AddTwoIntsRequest
from fredbots.srv import TaskAssign

env_row = 21
env_col = 21

# define actions
# numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

q_values = np.zeros((env_row, env_col, len(actions)))

# initialise rewards
rewards = np.full((env_row, env_col), -100.)


obstacles = [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
             [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [
                 8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20],
             [14, 0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9]]

# define training parameters
# the percentage of time when we should take the best action (instead of a random action)
epsilon = 0.9
discount_factor = 0.9  # discount factor for future rewards
learning_rate = 0.9  # the rate at which the agent should learn

x1 = 0
y1 = 0
z1 = 0
theta1 = 0


def construct_reward_matrix(goal_x, goal_y, obstacles=obstacles):
    rewards = np.full((env_row, env_col), -1)
    for obstacle in obstacles:
        try:
            # remove the reward for any actions that lead to the obstacle
            rewards[obstacle[0]][obstacle[1]] = -100
        except IndexError:
            continue
    rewards[goal_x][goal_y] = 100
    return rewards


def get_euclidian_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# Functions for QLearning

def is_final_state(row, col, goal_x, goal_y):
    if row == goal_x and col == goal_y:
        return True
    else:
        return False

# define a function that will choose a random, non-terminal starting location


def get_starting_location():
    # get a random row and column index
    current_row_index = np.random.randint(env_row)
    current_column_index = np.random.randint(env_col)

    return current_row_index, current_column_index

# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)


def get_next_action(current_row_index, current_column_index, epsilon):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:  # choose a random action
        return np.random.randint(4)

# define a function that will get the next location based on the chosen action


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

def get_next_random_location(current_row_index, current_column_index, action_index):
        
    new_row_index = current_row_index
    new_column_index = current_column_index

    while True:
        # Get random action index except the current action index
        random_action_index = np.random.randint(4)
        while random_action_index == action_index:
            random_action_index = np.random.randint(4)
        
        if actions[random_action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif actions[random_action_index] == 'right' and current_column_index < env_col - 1:
            new_column_index += 1
        elif actions[random_action_index] == 'down' and current_row_index < env_row - 1:
            new_row_index += 1
        elif actions[random_action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1

        if [new_row_index, new_column_index] not in obstacles:
            print("New location: ", new_row_index, new_column_index)
            time.sleep(1)
            break

    return new_row_index, new_column_index

# def reset_atom():
#     set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#     set_model_state_msg = ModelState()
#     set_model_state_msg.model_name = 'atom'
#     set_model_state_msg.pose.position.x = start_pos_x
#     set_model_state_msg.pose.position.y = start_pos_y
#     set_model_state_msg.pose.position.z = 0
#     set_model_state_msg.pose.orientation.x = 0
#     set_model_state_msg.pose.orientation.y = 0
#     set_model_state_msg.pose.orientation.z = 0
#     set_model_state_msg.pose.orientation.w = 1
#     set_model_state_proxy(set_model_state_msg)

# def stop_atom():
#     pub = rospy.Publisher('/atom/cmd_vel', Twist, queue_size=1)
#     twist = Twist()
#     pub.publish(twist)

# run through 1000 training episodes


def train(goal_x, goal_y, obstacles=obstacles):
    for episode in range(1000):
        # get the starting location for this episode
        row_index, column_index = get_starting_location()
        # continue taking actions (i.e., moving) until we reach a terminal state
        # (i.e., until we reach the item packaging area or crash into an item storage location)
        while not is_final_state(row_index, column_index, goal_x, goal_y):
            # choose which action to take (i.e., where to move next)
            action_index = get_next_action(row_index, column_index, epsilon)
            # perform the chosen action, and transition to the next state (i.e., move to the next location)
            # store the old row and column indexes
            old_row_index, old_column_index = row_index, column_index
            row_index, column_index = get_next_location(
                row_index, column_index, action_index)
            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index,
                                   old_column_index, action_index]
            temporal_difference = reward + \
                (discount_factor *
                 np.max(q_values[row_index, column_index])) - old_q_value
            # update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index,
                     action_index] = new_q_value
    print('Training complete!')

# Define a function that will get the shortest path between any location within the warehouse that
# the robot is allowed to travel and the item packaging location.


def get_shortest_path(start_row_index, start_column_index, goal_x, goal_y, atom_sim=False):
    # return immediately if this is an invalid starting location
    if is_final_state(start_row_index, start_column_index, goal_x, goal_y):
        return []
    else:  # if this is a 'legal' starting location
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        ql_control = True
        # continue moving along the path until we reach the goal (i.e., the item packaging location)
        while not is_final_state(current_row_index, current_column_index, goal_x, goal_y):

            rospy.wait_for_service('add_two_ints')
            add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)

            request = AddTwoIntsRequest()
            request.cur_x = current_row_index
            request.cur_y = current_column_index

            if ql_control:
                # get the best action to take
                action_index = get_next_action(
                    current_row_index, current_column_index, 1.)
                # move to the next location on the path, and add the new location to the list
                next_row_index, next_column_index = get_next_location(
                    current_row_index, current_column_index, action_index)
            
            request.next_x = next_row_index
            request.next_y = next_column_index

            if len(shortest_path) > 1:
                request.prev_x = shortest_path[-2][0]
                request.prev_y = shortest_path[-2][1]
            else:
                request.prev_x = -1
                request.prev_y = -1

            response = add_two_ints(request)

            if (response.occ == 0): # [NOT OCCUPIED]
                ql_control = True
                # print("occu: ", response.occ)
                if atom_sim:
                    # print("Atom_1 to location: ", next_row_index, next_column_index)
                    Goto_goal(next_row_index, next_column_index)

            else:
                ql_control = False
                action_index = get_next_action(
                    current_row_index, current_column_index, 1.)
                next_row_index, next_column_index = get_next_random_location(current_row_index, current_column_index, action_index)
                
                # current_row_index, current_column_index = next_row_index, next_column_index

                continue
                
            current_row_index, current_column_index = next_row_index, next_column_index

            shortest_path.append([current_row_index, current_column_index])
            # if atom_sim:
            #     print("Atom_3 moving to location: ",
            #           current_row_index, current_column_index)
            #     Goto_goal(current_row_index, current_column_index)
            # time.sleep(1)

        # Unlock the last location

        request = AddTwoIntsRequest()
        request.cur_x = current_row_index
        request.cur_y = current_column_index
        request.next_x = current_row_index
        request.next_y = current_column_index
        request.prev_x = shortest_path[-2][0]
        request.prev_y = shortest_path[-2][1]

        response = add_two_ints(request)
        
        # Write the shortest path to a text file
        file = open("shortest_path_3.txt", "a")
        file.write(str(shortest_path))
        file.write("\n")
        file.close()

        return shortest_path

# ROS movement functions and callbacks

# def laser_callback(msg):

#     laser_range = msg.ranges

#     coord = update_obst_coord(laser_range, msg.angle_min, msg.angle_max, len(laser_range))

#     approx_new_coord(coord)


def pose_callback(data):
    global x1, y1, z1, theta1
    x1 = data.pose.pose.position.x
    y1 = data.pose.pose.position.y

    # Change four quadrants data to first quadrant data
    x1 += 10
    y1 += 10

    roll, pitch, theta1 = euler_from_quaternion(
        [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])


def angle_difference(angle1, angle2):

    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)


def rotate(angular_speed_degree, relative_angle_degree, clockwise, goal_angle):

    velocity_message = Twist()
    velocity_message.linear.x = 0
    velocity_message.linear.y = 0
    velocity_message.linear.z = 0
    velocity_message.angular.x = 0
    velocity_message.angular.y = 0
    velocity_message.angular.z = 0

    angular_speed = math.radians(abs(angular_speed_degree))

    if (clockwise):
        velocity_message.angular.z = -abs(angular_speed)
    else:
        velocity_message.angular.z = abs(angular_speed)

    angle_moved = 0.0
    # we publish the velocity at 10 Hz (10 times a second)
    loop_rate = rospy.Rate(10)

    t0 = rospy.Time.now().to_sec()

    while True:
        # rospy.loginfo("Turtlesim rotates")
        pub.publish(velocity_message)

        t1 = rospy.Time.now().to_sec()
        current_angle_degree = (t1-t0)*angular_speed_degree
        loop_rate.sleep()

        # print 'current_angle_degree: ',current_angle_degree
        if angle_difference(math.degrees(theta1), goal_angle) < 1:
            # rospy.loginfo("Atom_3 rotated")
            break

    # finally, stop the robot when the distance is moved
    velocity_message.angular.z = 0
    pub.publish(velocity_message)


def Goto_goal(x_goal, y_goal):
    global x1, y1, theta1
    msg = Twist()

    distance = abs(math.sqrt(((x_goal-x1)**2)+((y_goal-y1)**2)))

    angle_t = math.atan2((y_goal-y1), (x_goal-x1))  # -pi to pi
    angle_t = math.degrees(angle_t)  # -180 to 180

    # if angle_t < 0:
    #     angle_t = 360 + angle_t

    theta1_deg = math.degrees(theta1)  # -180 to 180
    # angle = offset between current and goal angle ranging from -180 to 180
    if angle_t < 0:
        angle_t_b = 360 + angle_t  # 180 to 360
    else:
        angle_t_b = angle_t  # 0 to 180
    if theta1_deg < 0:
        theta1_deg_b = 360 + theta1_deg  # 180 to 360
    else:
        theta1_deg_b = theta1_deg  # 0 to 180

    angle = angle_t_b - theta1_deg_b  # -360 to 360

    # print(theta1_deg, angle_t, angle)
    phi = 1
    if -180 < angle < 0:
        rotate(min(abs(angle*phi), 30), abs(angle), True, angle_t)
    elif -360 < angle < -180:
        angle = 360 - abs(angle)
        rotate(min(abs(angle*phi), 30), abs(angle), False, angle_t)
    elif 180 < angle < 360:
        angle = 360 - angle
        rotate(min(abs(angle*phi), 30), angle, True, angle_t)
    else:
        rotate(min(abs(angle*phi), 30), angle, False, angle_t)

    while (distance > 0.15):
        Beta = 0.5
        distance = abs(math.sqrt(((x_goal-x1)**2)+((y_goal-y1)**2)))

        speed = distance*Beta

        msg.linear.x = speed
        msg.angular.z = 0

        pub.publish(msg)

    msg.linear.x = 0
    msg.angular.z = 0
    pub.publish(msg)

# def update_obst_coord(range_data, theta_min, theta_max, num_samples):

#     # Get the angle of each sample
#     thetas = np.linspace(theta_min, theta_max, num_samples)

#     # Adjust theta according to the orientation of the robot
#     thetas = theta1 + thetas

#     # Get the apparent x and y coordinates of each sample
#     xs, ys = [], []
#     for i in range(num_samples):
#         if range_data[i] < 5:
#             xs.append(range_data[i] * np.cos(thetas[i]))
#             ys.append(range_data[i] * np.sin(thetas[i]))
#         else:
#             xs.append(float('inf'))
#             ys.append(float('inf'))

#     # Calculate the lidar's offset from the robot's center
#     l_x = x1 + 0.2 * np.cos(theta1)
#     l_y = y1 + 0.2 * np.sin(theta1)

#     # Get the x and y coordinates of each sample
#     obst_coord = [(xs[i]+l_x, ys[i]+l_y) for i in range(num_samples)]

#     for (i,j) in obst_coord:
#       if i > env_col - 1 or i < 0 or j > env_row - 1 or j < 0:
#           obst_coord.remove((i,j))

#     return obst_coord

# def approx_new_coord(coord):
#     flag = 0
#     if not coord:
#         return
#     for (i,j) in coord:
#         if (i == float('inf')) or (i == float('-inf')) or (j == float('inf')) or (j == float('-inf')):
#             continue
#         else:
#             i_r = int(round(i,0))
#             j_r = int(round(j,0))
#             if [i_r,j_r] in obstacles_new:
#                 continue
#             else:
#                 obstacles_new.append([i_r,j_r])
#                 flag = 1
#     if flag == 1:
#         train()

def task_assigner(robot_id, x, y):
    rospy.wait_for_service('tasker')
    try:
        tasker = rospy.ServiceProxy('tasker', TaskAssign)
        resp = tasker(robot_id, x, y)
        return resp.robot_id, resp.a, resp.b, resp.x, resp.y
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None, None, None, None, None


def main():
    global pub, sub
    rospy.init_node("Shortest_path_atom_3")
    pub = rospy.Publisher("/atom_3/cmd_vel", Twist, queue_size=10)
    sub = rospy.Subscriber("/atom_3/odom", Odometry, pose_callback)
    # las = rospy.Subscriber("/scan_3", LaserScan, laser_callback)

    time.sleep(5)

    while not rospy.is_shutdown():

        robot_id_result, pickup_x, pickup_y, destination_x, destination_y = task_assigner(3, int(round(x1, 0)), int(round(y1, 0)))
        

        if robot_id_result == 3 and pickup_x != -1 and pickup_y != -1 and destination_x != -1 and destination_y != -1:
            print(f"Atom_3 assigned {pickup_x, pickup_y}, {destination_x, destination_y}")
            global rewards
            rewards = construct_reward_matrix(pickup_x, pickup_y)
            train(pickup_x, pickup_y)
            start_x, start_y = int(round(x1, 0)), int(round(y1, 0))
            print(f"Starting from {start_x, start_y}")
            get_shortest_path(start_x, start_y, pickup_x, pickup_y, atom_sim=True)
            print("Picked up")
            _, _, _, destination_x, destination_y = task_assigner(3, int(round(x1, 0)), int(round(y1, 0)))
            rewards = construct_reward_matrix(destination_x, destination_y)
            train(destination_x, destination_y)
            start_x, start_y = int(round(x1, 0)), int(round(y1, 0))
            print(f"Starting from {start_x, start_y}")
            get_shortest_path(start_x, start_y, destination_x, destination_y, atom_sim=True)
            print("Dropped off")

        time.sleep(1)

    time.sleep(5)


if __name__ == '__main__':
    main()
