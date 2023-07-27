#!/usr/bin/env python3

import time
from threading import Thread
import numpy as np
from tabulate import tabulate
import multiprocessing as mp
from fredbots.srv import TaskAssign
from fredbots.srv import TaskAssignResponse
from fredbots.srv import TaskAssignRequest

import rospy

class Robot:
    def __init__(self, robot_id, intital_position, battery_capacity=100):
        self.robot_id = robot_id
        self.current_position = intital_position
        self.batter_level = battery_capacity
        self.current_task = None
        self.is_idle = True
        self.destination_position = None

    def __str__(self):
        return f'Robot {self.robot_id}'

    def assign_package(self, package):
        self.current_task = package
        update_robot_task(self)
        self.destination_position = package.pickup_position

    def update_position(self, new_position):
        self.current_position = new_position

    def update_battery(self):
        if self.is_idle:
            self.batter_level -= 0.05
        elif self.current_task.weight > 0:
            self.batter_level -= 0.1 * self.current_task.weight/10
        else:
            self.batter_level -= 0.1


class Package:
    def __init__(self, package_id, pickup_position, dropoff_position, priority, weight=3):
        self.package_id = package_id
        self.pickup_position = pickup_position
        self.dropoff_position = dropoff_position
        self.weight = weight
        self.priority = priority
        self.picked_up = False
        self.delivered = False
        self.assigned = False
        self.utilities = {}

    def __str__(self):
        return f'Package {self.package_id}'

    def remove(self):
        self.picked_up = True
        self.delivered = True


def euclidean_distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



def assign_tasks(robots, packages):
    for package in packages:
        if not package.picked_up:
            max_distance = max([euclidean_distance(robot.current_position, package.pickup_position) for robot in robots])
            # store utility for each robot in package utility dictionary and sort by highest utility first  (descending order)
            package.utilities = {robot.robot_id: calculate_utility(robot, package, max_distance) for robot in robots}
            package.utilities = {k: v for k, v in sorted(package.utilities.items(), key=lambda item: item[1], reverse=True)}
    # packages.sort(key=lambda x: get_max_utility(x), reverse=True)
    # sort packages by first value in utilities dictionary (highest utility)
    packages.sort(key=lambda x: list(x.utilities.values())[0], reverse=True)
    assigned_robots = []
    for i in range(len(packages)):
        packages[i].utilities = {k: v for k, v in sorted(packages[i].utilities.items(), key=lambda item: item[1], reverse=True)}
        # packages.sort(key=lambda x: get_max_utility(x), reverse=True)
        packages.sort(key=lambda x: list(x.utilities.values())[0], reverse=True)
        for robot_id, utility in packages[i].utilities.items():
            if robot_id not in assigned_robots:
                robots[robot_id - 1].assign_package(packages[i])
                assigned_robots.append(robot_id)
                packages[i].assigned = True
                print(f'package {packages[i].package_id} assigned to robot {robot_id}')
                for other_package in packages:
                    if other_package.package_id != packages[i].package_id:
                        other_package.utilities[robot_id] = -1
                break
            else:
                packages[i].utilities[robot_id] = 0
    # return robots

 
def calculate_utility(robot, package, max_distance):
    distance = euclidean_distance(robot.current_position, package.pickup_position)
    utilty = (1 - distance/max_distance) * 0.3 + package.priority * 0.7
    # print(f'Utility of {robot} for package {package.package_id} is {utilty}')
    return utilty


def update_robot_position(robot, new_position):
    robot.update_position(new_position)


def update_robot_battery(robot):
    robot.update_battery()


def update_robot_task(robot):
    if robot.current_task is not None:
        if robot.current_task.picked_up and not robot.current_task.delivered:
            robot.is_idle = False
            robot.destination_position = robot.current_task.dropoff_position
        elif not robot.current_task.picked_up:
            robot.is_idle = False
            robot.destination_position = robot.current_task.pickup_position
    else:
        robot.is_idle = True
        robot.destination_position = None

def tasker_server():
    rospy.init_node('tasker_server')
    s = rospy.Service('tasker', TaskAssign, handle_tasker)
    print("Ready to assign tasks.")
    rospy.spin()

def handle_tasker(req):
    robot_id = req.robot_id
    x_pos = req.x
    y_pos = req.y

    # create robot object 
    robots = [Robot(robot_id, (x_pos, y_pos))]
    packages = [
        Package(1, (2, 2), (8, 8), 8, 5),
        Package(2, (5, 5), (3, 1), 4, 10),
        Package(3, (1, 1), (9, 9), 6, 2),
        Package(4, (7, 7), (4, 6), 2, 1),
    ]

    assign_tasks(robots, packages)

    # print(f"Robot {robot_id} is at position ({x_pos}, {y_pos})")
    robot_id = robots[0].robot_id
    x_pos = robots[0].destination_position[0]
    y_pos = robots[0].destination_position[1]
    print(f"Robot {robot_id} is supposed to go to ({x_pos}, {y_pos})")
    return robot_id, x_pos, y_pos



def main():
    packages = [
        Package(1, (2, 2), (8, 8), 8, 5),
        Package(2, (5, 5), (3, 1), 4, 10),
        Package(3, (1, 1), (9, 9), 6, 2),
        Package(4, (7, 7), (4, 6), 2, 1),
    ]

    # sort packages by priority
    packages.sort(key=lambda x: x.priority, reverse=True)

    # robots
    robots = [
        Robot(1, (0, 0)),
        Robot(2, (10, 10)),
        Robot(3, (5, 5))
    ]

    tasker_server() 

    """ # Start task assignment thread
    assign_thread = Thread(target=assign_tasks, args=(robots, packages))
    assign_thread.daemon = True
    assign_thread.start() """

    assign_tasks(robots, packages)

    # package and robot updates
    while True:
        # assign_tasks(robots, packages)
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
        for robot in robots:
            robot_table.append([
                robot.robot_id,
                robot.is_idle,
                robot.current_position,
                robot.destination_position,
                robot.current_task.package_id if robot.current_task is not None else "No task assigned",
            ])

        # Print table
        print("\n")
        print("Package Schedule:")
        print(tabulate(package_table, headers=[
              "ID", "Assigned", "Picked up", "Delivered", "Priority", "Pickup position", "Dropoff position"]))
        print("\nRobot Status:")
        print(tabulate(robot_table, headers=[
              "ID", "Idle", "Current position", "Destination position", "Current Task"]))

        time.sleep(2)  # Print updates every 2 seconds


if __name__ == "__main__":
    main()
