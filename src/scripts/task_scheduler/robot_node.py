import rospy

from fredbots.srv import *


def robot_client(robot_id, x, y):
    rospy.wait_for_service('tasker')
    try:
        tasker = rospy.ServiceProxy('tasker', TaskAssign)
        resp = tasker(robot_id, x, y)
        return resp.robot_id, resp.x, resp.y
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == "__main__":
    rospy.init_node('task_assign_client')
    
    robot_id = 1
    current_x = 10
    current_y = 20 

    robot_id_result, final_x, final_y = robot_client(robot_id, current_x, current_y)

    if robot_id_result is not None:
        rospy.loginfo(f"Received response from Robot {robot_id_result}: Final Position: ({final_x}, {final_y})")
    else:
        rospy.logerr("Failed to get a valid response.")
