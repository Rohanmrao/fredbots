#!/usr/bin/env python3

import time
from multiprocessing import Process
from threading import Thread

import numpy as np
import rospy
from tabulate import tabulate

from fredbots.srv import TaskAssign, TaskAssignRequest, TaskAssignResponse


class Robot:
    def __init__(self, robot_id, intital_position):
        self.robot_id = robot_id
        self.current_position = intital_position
        self.is_idle = True
        self.current_task = None
        self.task_positions = [None, None]

    def __str__(self):
        return f'Robot {self.robot_id}'

    def assign_package(self, package):
        self.current_task = package
        self.is_idle = False
        self.task_positions = [package.pickup_position, package.dropoff_position]


class Package:
    def __init__(self, package_id, pickup_position, dropoff_position, priority, weight=3):
        self.package_id = package_id
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        self.weight = weight
        self.priority = priority
        self.assigned = False
        self.picked_up = False
        self.delivered = False
        self.utilities = {}

    def __str__(self):
        return f'Package {self.package_id}'


def euclidean_distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def check_task_status(robots):
    for robot in robots.values():
        # print(robots, robot.current_task)
        if robot.current_task is not None:
            # print(robot.robot_id, robot.current_position, robot.task_positions[0]) if robot.robot_id == 1 else None
            # print(robot.robot_id, robot.current_position == robot.task_positions[0]) if robot.robot_id == 1 else None
            if robot.current_position == robot.task_positions[0]:
                robot.current_task.picked_up = True
                robot.task_positions[0] = None
            if robot.current_position == robot.task_positions[1]:
                robot.current_task.delivered = True
                robot.current_task = None
                robot.is_idle = True
                robot.task_positions[1] = None


def assign_tasks(robots, packages):
    # loop through all packages
    for package in packages:
        # if package has not been assigned
        if not package.assigned:
            # calculate the maximum distance between the package and all robots
            max_distance = max([euclidean_distance(robot.current_position, package.pickup_position)
                                for robot in robots.values()])
            # store utility for each robot in package utility dictionary and sort by highest utility first  (descending order)
            package.utilities = {robot.robot_id: calculate_utility(robot, package, max_distance) for robot in robots.values()}
            package.utilities = {k: v for k, v in sorted(package.utilities.items(), key=lambda item: item[1], reverse=True)}
            print(package, package.utilities)

    # sort packages by first value in utilities dictionary (highest utility)
    packages.sort(key=lambda x: list(x.utilities.values())[0], reverse=True)
    # loop through all packages
    for i in range(len(packages)):
        # sort the utilities of the current package by highest utility first (descending order)
        packages[i].utilities = {k: v for k, v in sorted(
            packages[i].utilities.items(), key=lambda item: item[1], reverse=True)}
        # sort packages by first value in utilities dictionary (highest utility)
        packages.sort(key=lambda x: list(x.utilities.values())[0], reverse=True)
        # loop through all robots in the utilities dictionary of the current package
        print(packages[i].utilities)
        for robot_id, utility in packages[i].utilities.items():
            # if the robot has not been assigned a package
            if robots[robot_id].is_idle and not packages[i].assigned:
                # assign the package to the robot
                robots[robot_id].assign_package(packages[i])
                packages[i].assigned = True
                # update the utilities of all other packages for the assigned robot to -1
                for other_package in packages:
                    if other_package.package_id != packages[i].package_id:
                        other_package.utilities[robot_id] = -1
                break
            else:
                # if the robot has already been assigned a package, set the utility of the robot for the current package to 0
                packages[i].utilities[robot_id] = 0
    # return robots


def calculate_utility(robot, package, max_distance):
    distance = euclidean_distance(robot.current_position, package.pickup_position)
    utilty = (1 - distance/max_distance) * 0.3 + package.priority * 0.7
    return utilty



def fetch_packages():
    global packages, package_number
    try:
        file = open("package.txt", "r")
        lines = file.readlines()
        # print(lines)
        for line in lines:
            package_number += 1
            package_info = line.split()
            package_id = package_number
            pickup_x = int(package_info[0].strip())
            pickup_y = int(package_info[1].strip())
            dropoff_x = int(package_info[2].strip())
            dropoff_y = int(package_info[3].strip())
            priority = int(package_info[4].strip())

            packages.append(Package(package_id, (pickup_x, pickup_y), (dropoff_x, dropoff_y), priority))

            # delete the first line in the file
            lines.pop(0)

        # write the remaining lines back to the file
        file = open("package.txt", "w")
        for line in lines:
            file.write(line)

            
        file.close()
    except:
        pass

    # sort packages by priority
    packages.sort(key=lambda x: x.priority, reverse=True)




def tasker_server():
    rospy.init_node('tasker_server')
    s = rospy.Service('tasker', TaskAssign, handle_tasker)
    rospy.loginfo("Tasker server ready.")
    rospy.spin()


def handle_tasker(req):
    global robots
    robot_id = req.robot_id
    x_pos = req.x
    y_pos = req.y

    if robot_id in robots:
        robots[robot_id].current_position = (x_pos, y_pos)
    else:
        robots[robot_id] = Robot(robot_id, (x_pos, y_pos))


    # fetch packages
    fetch_packages()
    # print(len(packages))

    # task status
    check_task_status(robots)

    # assign tasks
    assign_tasks(robots, packages)

    # fetch robot task position
    try:
        pickup_x, pickup_y = robots[robot_id].task_positions[0]
        destination_x, destination_y = robots[robot_id].task_positions[1]
    except:
        pickup_x, pickup_y = None, None
        destination_x, destination_y = None, None

    # Table

    package_table = []
    for package in packages:
        package_table.append([
            package.package_id,
            package.assigned,
            package.picked_up,
            package.delivered,
            package.priority,
            package.pickup_position,
            package.dropoff_position
        ])

    robot_table = []
    for robot in robots.values():
        robot_table.append([
            robot.robot_id,
            robot.is_idle,
            robot.current_position,
            robot.current_task.package_id if robot.current_task is not None else "No task assigned",
            robot.task_positions[0],
            robot.task_positions[1],
        ])

    print("Package Schedule:")
    print(tabulate(package_table, headers=["ID", "Assigned", "Picked up",
          "Delivered", "Priority", "Pickup position", "Dropoff position"]))

    print("Robot Status:")
    print(tabulate(robot_table, headers=["ID", "Idle", "Current position",
          "Current Task", "Pickup position", "Dropoff position"]))

    return robot_id, pickup_x, pickup_y, destination_x, destination_y


def main():
    global robots, packages, package_number 
    packages = []
    package_number = 0

    # robots
    robots = dict()

    tasker_server()

if __name__ == '__main__':
    main()