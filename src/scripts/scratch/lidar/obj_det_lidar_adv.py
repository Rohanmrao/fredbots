import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from math import atan2

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Global variables
x = 0.0
y = 0.0
theta = 0.0

obst_coord = []
obstacles = {}

# Callback function for odometry
def newOdom(msg):
    global x
    global y
    global theta

    # Get the position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    # Get the orientation
    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

def update_obst_coord(range_data, theta_min, theta_max, num_samples):
    global obst_coord

    # Get the angle of each sample
    thetas = np.linspace(theta_min, theta_max, num_samples)

    # print(thetas[:10], thetas[-10:])

    # Adjust theta according to the orientation of the robot
    thetas = theta + thetas

    # print(thetas[:10], thetas[-10:])

    # Get the apparent x and y coordinates of each sample
    # xs = [range_data[i] * np.cos(thetas[i]) for i in range(num_samples) if range_data[i] < 10 else float('inf')]
    # ys = [range_data[i] * np.sin(thetas[i]) for i in range(num_samples) if range_data[i] < 10 else float('inf')]
    xs, ys = [], []
    for i in range(num_samples):
        if range_data[i] < 5:
            xs.append(range_data[i] * np.cos(thetas[i]))
            ys.append(range_data[i] * np.sin(thetas[i]))
        else:
            xs.append(float('inf'))
            ys.append(float('inf'))

    # Calculate the lidar's offset from the robot's center
    l_x = x + 0.2 * np.cos(theta)
    l_y = y + 0.2 * np.sin(theta)

    # Get the x and y coordinates of each sample
    obst_coord = [(xs[i]+l_x, ys[i]+l_y) for i in range(num_samples)]

    approx_new_coord(obst_coord)
    

    # # Get the x and y coordinates of each sample
    # xs = [range_data[i] * np.cos(thetas[i]) for i in range(num_samples)]
    # ys = [range_data[i] * np.sin(thetas[i]) for i in range(num_samples)]

    # # Get the x and y coordinates of each sample
    # obst_coord = [(xs[i], ys[i]) for i in range(num_samples)]



# Callback function for lidar
def scan_callback(msg):
    # Get the distance d
    ds = msg.ranges

    thetas = np.linspace(msg.angle_min, msg.angle_max, len(ds))

    # Get the number of samples
    num_samples = len(ds)

    # Get the angle of the sample that is closest to the robot
    theta_min = msg.angle_min

    # Get the angle of the sample that is farthest from the robot
    theta_max = msg.angle_max

    # Update the coordinates of the obstacles
    update_obst_coord(ds, theta_min, theta_max, num_samples)


def approx_new_coord(coord):
    flag = 0
    for (i,j) in coord:
        if (i == float('inf')) or (i == float('-inf')) or (j == float('inf')) or (j == float('-inf')):
            continue
        else:
            i_r = int(round(i,0))
            j_r = int(round(j,0))
            if ((i_r,j_r) in obstacles.keys()) and obstacles[(i_r,j_r)] < 10:
                obstacles[(i_r,j_r)] += 1
                continue
            else:
                obstacles[(i_r,j_r)] = 1
            print(obstacles)


def main():
    # Initialize node
    rospy.init_node('obstacle_detector')

    # Create a subscriber to the topic /odom
    sub = rospy.Subscriber('/atom/odom', Odometry, newOdom)

    # Create a subscriber to the topic /scan
    sub = rospy.Subscriber('/scan', LaserScan, scan_callback)

    # Set the rate
    rate = rospy.Rate(0.1)

    # Create a Twist message
    vel = Twist()

    # Set the linear velocity
    vel.linear.x = 0.5

    # Set the angular velocity
    vel.angular.z = 0.0

    # While ROS is still running
    while not rospy.is_shutdown():
        # Publish the message
        # pub.publish(vel)

        # Sleep for the rest of the cycle
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass