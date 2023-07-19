import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_srvs.srv import Empty
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TurtleBot3Controller:
    def __init__(self):
        self.state = None
        # self.target_x = None
        # self.target_y = None

        self.target_x = 4
        self.target_y = 4
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
        self.reset_proxy = rospy.ServiceProxy("/reset", Empty)

        self.rate = rospy.Rate(10)  # 10hz 

    def pose_callback(self, data):
        self.state = [
            data.x,
            data.y,
            data.theta,
            data.linear_velocity,
            data.angular_velocity,
        ]
    
    def reset_turtlesim(self):
        rospy.wait_for_service("/reset")
        try:
            reset_service = rospy.ServiceProxy("/reset", Empty)
            reset_service()
            rospy.sleep(1.0)
        except rospy.ServiceException as e:
            print("Reset service call failed:", str(e))
    
    def move(self, linear_vel, angular_vel):
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_vel
        velocity_msg.angular.z = angular_vel
        self.velocity_publisher.publish(velocity_msg)

    def move_turtle(self, action):
        if action == 0:  # Up
            self.move(3.0, 0.0)
        elif action == 1:  # Down
            self.move(-3.0, 0.0)
        elif action == 2:  # Left
            self.move(0.0, 3.0)
        elif action == 3:  # Right
            self.move(0.0, -3.0)

    def is_done(self):
        current_x, current_y, current_theta, _, _ = self.state

        distance_to_target = self.euclidean_distance(
                        current_x, current_y, self.target_x, self.target_y
                    )
        print("distance to target: ", distance_to_target)

        distance_to_bound = self.euclidean_distance(
                        current_x, current_y, 5.544445, 5.544445
                    )
        print("distance_to_bound: ", distance_to_bound)
        
        if distance_to_bound >3.5:
            return True
        
        if distance_to_target < 0.5:
            print("REACHED GOAL!!!!!!!!!!!!!!!!!!!!")
            return True
        
        return False

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


    def test(self):
        # Use the model to predict actions
        self.reset_turtlesim()
        current_x, current_y, current_theta, _, _ = self.state

        distance_to_target = self.euclidean_distance(
                        current_x, current_y, self.target_x, self.target_y
                    )
        
        target_angle = math.atan2(self.target_y - current_y, self.target_x - current_x)
        # print("target angle b4: ", target_angle)
        if target_angle < 0:
            target_angle += 2 * math.pi

        
        relative_angle = target_angle - current_theta
        if relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        elif relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        current_theta = current_theta if current_theta >= 0 else 2 * math.pi + current_theta
        
        state = np.array([current_x, current_y, current_theta, distance_to_target, relative_angle])
        done = False

        while not done:
            # Predict action probabilities and value

            logits, value = loaded_model(tf.expand_dims(state, axis=0))

            # Sample action from policy logits
            action = tf.random.categorical(logits, num_samples=1).numpy()[0, 0]

            # Execute action
            self.move_turtle(action)
            next_state = self.state
            # next_state, reward, done = env.step(action)
            done = self.is_done()

            state = next_state




if __name__ == "__main__":
    # Load the saved model
    loaded_model = tf.keras.models.load_model('model')

    test = TurtleBot3Controller()
    test.test()

    # Use the model to predict actions
    # state = env.reset()
    # done = False

    # while not done:
    #     # Predict action probabilities and value
    #     logits, value = loaded_model(tf.expand_dims(state, axis=0))

    #     # Sample action from policy logits
    #     action = tf.random.categorical(logits, num_samples=1).numpy()[0, 0]

    #     # Execute action
    #     next_state, reward, done = env.step(action)

    #     state = next_state




