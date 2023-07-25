import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class TurtleBot3Controller:
    def __init__(self):
        self.state = None
        # self.target_x = None
        # self.target_y = None

        self.target_x = 4
        self.target_y = 4

        self.count = 0

        self.model = load_model('model.h5')
        # actions:
        # Move forward with a moderate linear velocity.
        # Move backward with a moderate linear velocity.
        # Rotate clockwise with a moderate angular velocity.
        # Rotate counterclockwise with a moderate angular velocity.
        # Stop, maintaining the current linear and angular velocities.

        rospy.init_node("turtlebot_controller", anonymous=True)
        self.velocity_publisher = rospy.Publisher(
            "/turtle1/cmd_vel", Twist, queue_size=10
        )
        self.pose_subscriber = rospy.Subscriber(
            "/turtle1/pose", Pose, self.pose_callback
        )
        # self.reset_proxy = rospy.ServiceProxy("/reset", Empty)

        self.rate = rospy.Rate(10)  # 10hz

    def pose_callback(self, data):
        self.state = [
            data.x,
            data.y,
            data.theta,
            data.linear_velocity,
            data.angular_velocity,
        ]

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_target_position(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def move_turtle(self, action):
        if action == 0:  # Up
            self.move(3.0, 0.0)
        elif action == 1:  # Down
            self.move(-3.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 3.0)
        elif action == 3:  # Right
            self.move(0.0, -3.0)

    def move(self, linear_vel, angular_vel):
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_vel
        velocity_msg.angular.z = angular_vel
        self.velocity_publisher.publish(velocity_msg)
    
    def shortest_path(self):
        self.set_target_position(4, 4)

        while not rospy.is_shutdown():
            if self.state is not None:
                state = self.state

                current_x, current_y, current_theta, _, _ = self.state
                distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)

                if distance_to_target < 0.5:  # Reached the target
                    print("Target reached!")
                    break

                action = self.model.predict(state)
                print("action: ",action)
                self.move_turtle(action)

                self.rate.sleep()



if __name__ == '__main__':
    # print("tf version: ", tf.version())
    controller = TurtleBot3Controller()
    controller.shortest_path()
