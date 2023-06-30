#!/usr/bin/env python3
import rospy
import yaml
import math
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist

x1 = 0
y1 = 0
z1 = 0
theta1 = 0

       

def pose_callback(pose_msg:Pose):
    global x1,y1,z1,theta1
    x1 = pose_msg.x 
    y1 = pose_msg.y 
    theta1 = pose_msg.theta 


def Goto_goal(x_goal, y_goal):
    global x1,y1,theta1
    msg = Twist()

    while(True):
        Beta = 0.9
        distance = abs(math.sqrt(((x_goal-x1)**2)+((y_goal-y1)**2)))

        speed = distance*Beta

        Phi = 4.0
        ang_dist = math.atan2((y_goal-y1),(x_goal-x1))

        ang_speed = Phi*(ang_dist-theta1)

        msg.linear.x = speed
        msg.angular.z = ang_speed

        pub.publish(msg)

        if (distance<0.01):
            break




if __name__ == "__main__":

        rospy.init_node("GoTo_Controller")
        pub = rospy.Publisher("/turtle1/cmd_vel",Twist, queue_size =10)
        sub = rospy.Subscriber("/turtle1/pose",Pose, callback = pose_callback)
        rospy.loginfo("Node has been started")

        userX = float(input("Please enter X goal"))
        userY = float(input("Please enter Y goal"))

        Goto_goal(userX,userY)
        print(x1,y1,theta1)