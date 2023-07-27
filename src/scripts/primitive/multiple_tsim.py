import rospy
from turtlesim.srv import Spawn
from turtlesim.msg import Pose

def spawn_turtle(name, x, y, theta):
    rospy.wait_for_service('/spawn')
    try:
        spawn_turtle = rospy.ServiceProxy('/spawn', Spawn)
        response = spawn_turtle(x, y, theta, name)
        return response.name
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

rospy.init_node('multiple_turtles')

# Spawn Turtles
turtle1_name = spawn_turtle("turtle1", 2, 2, 0)
turtle2_name = spawn_turtle("turtle2", 5, 5, 0)
turtle3_name = spawn_turtle("turtle3", 8, 8, 0)

# Print the spawned turtle names
print("Spawned Turtles:")
print("Turtle 1:", turtle1_name)
print("Turtle 2:", turtle2_name)
print("Turtle 3:", turtle3_name)

# Continue with other operations or control the turtles
