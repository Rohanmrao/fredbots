#!/usr/bin/env python

import rospy
import yaml
from nav_msgs.msg import MapMetaData, OccupancyGrid
import matplotlib.pyplot as plt
import numpy as np
import imageio

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

    # Assuming you have already imported the necessary ROS modules and defined the map_metadata_msg and map_msg

    # Assuming you have already imported the necessary ROS modules and defined the occupancy_grid_msg

    coords = []

    print("\n",map_metadata_msg,"\n")

    for width in range(0, map_metadata_msg.width):
        for height in range(0, map_metadata_msg.height):
            if map_metadata_msg == 0:
                x = width * map_metadata_msg.resolution + map_metadata_msg.resolution / 2
                y = height * map_metadata_msg.resolution + map_metadata_msg.resolution / 2

                coords.append([x,y])

    print("\n")
    print(coords)
    print(len(coords))



    return map_metadata_msg

# def publish_occupancy_grid(map_metadata_msg):

#     occupancy_grid_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)

#     occupancy_grid_msg = OccupancyGrid()

#     # Load map data from the PGM file
#     with open('/home/rohan/catkin_ws/src/fredbots/maps/mymap.pgm', 'r') as f:
#         map_data = f.read()

#     # Set occupancy grid attributes
#     occupancy_grid_msg.header.stamp = rospy.Time.now()
#     occupancy_grid_msg.header.frame_id = 'map'
#     occupancy_grid_msg.info.map_load_time = rospy.Time.now()
#     occupancy_grid_msg.info.resolution = yaml_data['resolution']
#     occupancy_grid_msg.info.width = yaml_data['width']
#     occupancy_grid_msg.info.height = yaml_data['height']
#     occupancy_grid_msg.info.origin = map_metadata_msg.origin

#     # Convert bytes to integers and create the data list for the OccupancyGrid message
#     occupancy_grid_msg.data = list(map(int, map_data))

#     rospy.sleep(1)  # Give some time for subscribers to connect
#     occupancy_grid_pub.publish(occupancy_grid_msg)
#     rospy.loginfo("Published OccupancyGrid message:")
#     rospy.loginfo(occupancy_grid_msg)

pub = rospy.Publisher('occupancy_grid', OccupancyGrid, queue_size=10)
occupancy_grid = OccupancyGrid()

def read_map_files(pgm_file_path, yaml_file_path):
    # Read the PGM file using imageio
    map_data = imageio.imread(pgm_file_path)

    # Read the YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        map_metadata = yaml.safe_load(yaml_file)

    return map_data, map_metadata


def plot_occupancy_grid(map_data, map_metadata):
    # Convert map data to a numpy array
    map_array = np.array(map_data, dtype=np.int8)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the occupancy grid using imshow
    cmap = plt.cm.binary  # You can choose a different colormap if needed
    im = ax.imshow(map_array, cmap=cmap, origin='lower', extent=[0, map_metadata['width'] * map_metadata['resolution'],
                                                                 0, map_metadata['height'] * map_metadata['resolution']])

    # Add labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Occupancy Grid')

    # Add a colorbar legend
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy Value')

    # Show the plot
    plt.show()

def read_map_data_from_pgm(pgm_file_path):

    # Read the PGM file using imageio
    map_data = imageio.imread(pgm_file_path)

    # Convert the grayscale values to occupancy grid values (0, 100)
    occupancy_grid_values = (map_data.astype(np.float32) / 255) * 100

    return occupancy_grid_values


if __name__ == '__main__':

    
    print("\n*************\n")
    map_meta = publish_map_metadata()
    print("\n*************\n")
    #publish_map_meta(map_meta)

    pgm_file_path = '/home/rohan/catkin_ws/src/fredbots/maps/mymap.pgm'
    yaml_file_path = '/home/rohan/catkin_ws/src/fredbots/maps/mymap.yaml'

    map_data, map_metadata = read_map_files(pgm_file_path, yaml_file_path)

    occupancy_grid_values = read_map_data_from_pgm(pgm_file_path)
    print(occupancy_grid_values[10][942])

    plot_occupancy_grid(map_data,map_metadata)

