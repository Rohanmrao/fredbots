<<<<<<< HEAD
<<<<<<< HEAD
import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np
import tensorflow as tf

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        actor = tf.keras.models.Sequential()
        actor.add(tf.keras.layers.Dense(32, input_dim=self.state_size, activation='relu'))
        actor.add(tf.keras.layers.Dense(32, activation='relu'))
        actor.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
        return actor

    def build_critic(self):
        critic = tf.keras.models.Sequential()
        critic.add(tf.keras.layers.Dense(32, input_dim=self.state_size, activation='relu'))
        critic.add(tf.keras.layers.Dense(32, activation='relu'))
        critic.add(tf.keras.layers.Dense(1, activation='linear'))
        return critic

    def load_model_weights(self, weights_file):
        self.actor.load_weights(weights_file)
        self.critic.load_weights(weights_file)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        return np.argmax(self.actor.predict(state)[0])


class TurtleBot3Controller:
    def __init__(self):
        rospy.init_node('turtlebot3_controller', anonymous=True)
        self.state = None
        self.target_x = 0
        self.target_y = 0
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/pose', Pose, self.pose_callback)
        self.reset_proxy = rospy.ServiceProxy('/reset', Empty)
        self.rate = rospy.Rate(10)  # 10hz
        self.agent = ActorCriticAgent(state_size=5, action_size=4)

    # ... Rest of the code

    def shortest_path(self, weights_file):
        self.agent.load_model_weights(weights_file)
        self.set_target_position(4, 4)

        while not rospy.is_shutdown():
            if self.state is not None:
                state = self.state

                current_x, current_y, current_theta, _, _ = self.state
                distance_to_target = self.euclidean_distance(current_x, current_y, self.target_x, self.target_y)

                if distance_to_target < 0.5:  # Reached the target
                    print("Target reached!")
                    break

                action = self.agent.get_action(state)
                self.move_turtle(action)

                self.rate.sleep()

        # Stop the turtle
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(vel_msg)

        self.reset_turtlesim()


# Example usage
if __name__ == '__main__':
    print("tf version: ", tf.version())
    controller = TurtleBot3Controller()
    controller.shortest_path(weights_file='model.h5')
=======
>>>>>>> parent of 1fdab71... model
=======
>>>>>>> parent of 1fdab71... model
