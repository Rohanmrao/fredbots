import rospy
from sensor_msgs.msg import LaserScan

def laser_scan_callback(msg):
    # Access the range measurements from the LaserScan message
    ranges = msg.ranges

    # Find the minimum and maximum range values
    min_range = min(ranges)
    max_range = max(ranges)
    

    print("mini range: ", min_range)



    # Print the results
    # rospy.loginfo("Minimum range: %.2f meters" % min_range)
    # rospy.loginfo("Maximum range: %.2f meters" % max_range)

rospy.init_node('laser_scan_subscriber')
rospy.Subscriber('/scan', LaserScan, laser_scan_callback)
rospy.spin()
