#!/usr/bin/env python

import rospy
from fredbots.srv import AddTwoInts
from fredbots.srv import AddTwoIntsRequest

rospy.init_node('add_two_ints_client')
rospy.wait_for_service('add_two_ints')
add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)

request = AddTwoIntsRequest()
request.cur_x = 1
request.cur_y = 1

request.next_x = 1
request.next_y = 2

response = add_two_ints(request)
print("occupancy:", response.occ)
