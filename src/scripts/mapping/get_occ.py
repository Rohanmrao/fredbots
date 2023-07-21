#!/usr/bin/env python

import rospy
import yaml
from nav_msgs.msg import MapMetaData, OccupancyGrid

rospy.init_node('occupancy_metadata_grid_publisher', anonymous=True)

# Load map data from the YAML file
with open('/home/rohan/catkin_ws/src/fredbots/maps/mymap.yaml', 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

def publish_map_metadata():
    map_metadata_pub = rospy.Publisher('/map_metadata', MapMetaData, queue_size=10)

    map_metadata_msg = MapMetaData()

    # Set map metadata attributes
    map_metadata_msg.map_load_time = rospy.Time.now()
    map_metadata_msg.resolution = yaml_data['resolution']
    map_metadata_msg.width = yaml_data['width']
    map_metadata_msg.height = yaml_data['height']

    # You may need to adjust these values according to your YAML file
    map_metadata_msg.origin.position.x = yaml_data['origin'][0]
    map_metadata_msg.origin.position.y = yaml_data['origin'][1]
    map_metadata_msg.origin.position.z = 0.0

    map_metadata_msg.origin.orientation.x = 0.0
    map_metadata_msg.origin.orientation.y = 0.0
    map_metadata_msg.origin.orientation.z = 0.0
    map_metadata_msg.origin.orientation.w = 1.0

    rospy.sleep(1)  # Give some time for subscribers to connect
    map_metadata_pub.publish(map_metadata_msg)
    rospy.loginfo("Published MapMetaData message:")
    rospy.loginfo(map_metadata_msg)

    return map_metadata_msg

def publish_occupancy_grid(map_metadata_msg):

    occupancy_grid_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)

    occupancy_grid_msg = OccupancyGrid()

    # Load map data from the PGM file
    with open('/home/rohan/catkin_ws/src/fredbots/maps/mymap.pgm', 'r') as f:
        map_data = f.read()

    # Set occupancy grid attributes
    occupancy_grid_msg.header.stamp = rospy.Time.now()
    occupancy_grid_msg.header.frame_id = 'map'
    occupancy_grid_msg.info.map_load_time = rospy.Time.now()
    occupancy_grid_msg.info.resolution = yaml_data['resolution']
    occupancy_grid_msg.info.width = yaml_data['width']
    occupancy_grid_msg.info.height = yaml_data['height']
    occupancy_grid_msg.info.origin = map_metadata_msg.origin

    # Convert bytes to integers and create the data list for the OccupancyGrid message
    occupancy_grid_msg.data = list(map(int, map_data))

    rospy.sleep(1)  # Give some time for subscribers to connect
    occupancy_grid_pub.publish(occupancy_grid_msg)
    rospy.loginfo("Published OccupancyGrid message:")
    rospy.loginfo(occupancy_grid_msg)

if __name__ == '__main__':
    try:
        map_meta = publish_map_metadata()
        print("\n*************\n")
        #publish_occupancy_grid(map_meta)
    except rospy.ROSInterruptException:
        pass
