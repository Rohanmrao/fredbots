#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math


class BotDrive:
    def __init__(self):
        self.nh_priv = rospy.get_namespace()
        rospy.loginfo("BotDrive Simulation Node Init")
        ret = self.init()
        assert ret

    def __del__(self):
        self.update_command_velocity(0.0, 0.0)
        rospy.signal_shutdown("Shutdown Requested")

    def init(self):

        self.escape_range = 30.0 * math.pi / 180.0
        self.check_forward_dist = 0.7
        self.check_side_dist = 0.6

        self.tb3_pose = 0.0
        self.prev_tb3_pose = 0.0

        self.scan_data = [0.0, 0.0, 0.0]

        self.cmd_vel_pub = rospy.Publisher('/atom_4/cmd_vel', Twist, queue_size=10)

        rospy.Subscriber("/scan_4", LaserScan, self.laser_scan_msg_callback)
        rospy.Subscriber('/atom_4/odom', Odometry, self.odom_msg_callback)

        return True

    def odom_msg_callback(self, msg):
        siny = 2.0 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y)
        cosy = 1.0 - 2.0 * (msg.pose.pose.orientation.y * msg.pose.pose.orientation.y + msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)

        self.tb3_pose = math.atan2(siny, cosy)

    def laser_scan_msg_callback(self, msg):
        scan_angle = [0, 30, 330]
        self.scan_data = [0.0, 0.0, 0.0]

        for num in range(3):
            if math.isinf(msg.ranges[scan_angle[num]]):
                self.scan_data[num] = msg.range_max
            else:
                self.scan_data[num] = msg.ranges[scan_angle[num]]

    def update_command_velocity(self, linear, angular):
        cmd_vel = Twist()
        cmd_vel.linear.x = linear
        cmd_vel.angular.z = angular
        self.cmd_vel_pub.publish(cmd_vel)

    def control_loop(self):
        bot_state_num = 0

        if bot_state_num == 0:
            if self.scan_data[1] > self.check_forward_dist:
                if self.scan_data[0] < self.check_side_dist:
                    self.prev_tb3_pose = self.tb3_pose
                    bot_state_num = 2
                elif self.scan_data[2] < self.check_side_dist:
                    self.prev_tb3_pose = self.tb3_pose
                    bot_state_num = 3
                else:
                    bot_state_num = 1

            if self.scan_data[1] < self.check_forward_dist:
                self.prev_tb3_pose = self.tb3_pose
                bot_state_num = 2

        elif bot_state_num == 1:
            self.update_command_velocity(0.2, 0.0)
            bot_state_num = 0

        elif bot_state_num == 2:
            if abs(self.prev_tb3_pose - self.tb3_pose) >= self.escape_range:
                bot_state_num = 0
            else:
                self.update_command_velocity(0.0, -1.0)
                
        elif bot_state_num == 3:
            if abs(self.prev_tb3_pose - self.tb3_pose) >= self.escape_range:
                bot_state_num = 0
            else:
                self.update_command_velocity(0.0, 1.0)

        else:
            bot_state_num = 0

        return True


if __name__ == "__main__":
    rospy.init_node("bot_drive")
    bot_drive = BotDrive()
    loop_rate = rospy.Rate(125)

    while not rospy.is_shutdown():
        bot_drive.control_loop()
        rospy.spinOnce()
        loop_rate.sleep()
