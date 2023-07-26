#!/usr/bin/env python

import rospy
from fredbots.srv import LocalCtrl
from fredbots.srv import LocalCtrlRequest

rospy.init_node('add_two_ints_client')
rospy.wait_for_service('add_two_ints')
add_two_ints = rospy.ServiceProxy('add_two_ints', LocalCtrl)

request = LocalCtrlRequest()
request.cur_x = 1
request.cur_y = 2

request.next_x = 2
request.next_y = 2

response = add_two_ints(request)
print("occupancy:", response.occ)
