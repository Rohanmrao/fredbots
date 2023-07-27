import rospy
from fredbots.srv import TaskAssign

def robot_client(robot_id, x, y):
    rospy.wait_for_service('tasker')
    try:
        tasker = rospy.ServiceProxy('tasker', TaskAssign)
        resp = tasker(robot_id, x, y)
        return resp.robot_id, resp.a, resp.b, resp.x, resp.y
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None, None, None, None, None

if __name__ == "__main__":
    rospy.init_node("robot4_client")

    robot_id = 4
    current_x = 9
    current_y = 3

    while not rospy.is_shutdown():
        robot_id_result, pickup_x, pickup_y, destination_x, destination_y = robot_client(robot_id, current_x, current_y)
        print("id, pickup_x, pickup_y: ", robot_id_result, pickup_x, pickup_y, destination_x, destination_y)

        if robot_id_result is not None:
            rospy.loginfo(f"Received response from Robot {robot_id_result}: Pickup Position: ({pickup_x}, {pickup_y}), Destination Position: ({destination_x}, {destination_y})")
        else:
            rospy.logerr("Failed to get a valid response.")

        rospy.sleep(1)