# Harvesting lidar data from the robot sensor

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import atan2

# Global variables
x = 0.0
y = 0.0
theta = 0.0

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

    plt.figure(1)
    plt.clf()
    plt.polar(thetas, ds, 'ro')
    plt.show()


# Main function
def main():
    # Initialize the node
    rospy.init_node('lidar_process')

    # Subscribe to the odometry topic
    sub_odom = rospy.Subscriber('/odom', Odometry, newOdom)

    # Subscribe to the lidar topic
    sub_scan = rospy.Subscriber('/scan', LaserScan, scan_callback)

    # Set the rate
    rate = rospy.Rate(20)

    # Main loop
    while not rospy.is_shutdown():
        # Print the position
        print(x, y, theta)

        # Sleep
        rate.sleep()

# Main function call
if __name__ == '__main__':
    main()