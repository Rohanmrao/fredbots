#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

def move_in_circle():
    rospy.init_node('move_in_circle', anonymous=True)
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Create a Twist message and set linear and angular velocities
    move_cmd = Twist()
    move_cmd.linear.x = 1.0  # linear velocity along the x-axis
    move_cmd.angular.z = 0.5  # angular velocity around the z-axis

    while not rospy.is_shutdown():
        velocity_publisher.publish(move_cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_in_circle()
    except rospy.ROSInterruptException:
        pass
