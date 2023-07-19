import time
from threading import Thread

import numpy as np
from tabulate import tabulate


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
        self.is_idle = False
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
            robot = find_max_utlity_robot(robots, package)
            if robot is not None:
                robot.assign_package(package)
                package.picked_up = True
                print(f'{robot} will pick up package {package.package_id}')
            else :
                print('No robots available') 


def find_max_utlity_robot(robots, package):
    # use a utility function to find the best robot use distace/max distance of all robots
    max_utility = float('-inf')
    best_robot = None
    max_distance = max([euclidean_distance(
        robot.current_position, package.pickup_position) for robot in robots])
    for robot in robots:
        if robot.current_task is None:
            utility = calculate_utility(robot, package, max_distance)
            if utility > max_utility:
                max_utility = utility
                best_robot = robot
    try:
        return best_robot
    except:
        print('No robots available')
        return None


def calculate_utility(robot, package, max_distance):
    distance = euclidean_distance(
        robot.current_position, package.pickup_position)
    return (1 - distance/max_distance) * 0.5 + package.priority * 0.5


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


def main():
    packages = [
        Package(1, (2, 2), (8, 8), 2, 5),
        Package(2, (5, 5), (3, 1), 3, 10),
        Package(3, (1, 1), (9, 9), 1, 2),
        Package(4, (7, 7), (4, 6), 4, 1),
    ]

    # sort packages by priority
    packages.sort(key=lambda x: x.priority, reverse=False)

    # robots
    robots = [
        Robot(1, (0, 0)),
        Robot(2, (10, 10)),
        Robot(3, (5, 5))
    ]

# Start task assignment thread
    assign_thread = Thread(target=assign_tasks, args=(robots, packages))
    assign_thread.daemon = True
    assign_thread.start()

    # package and robot updates
    while True:
        package_table = []
        for package in packages:
            package_table.append([
                package.package_id,
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
                robot.current_task.package_id if robot.current_task is not None else None,
            ])

        # Print table
        print("\n")
        print("Package Schedule:")
        print(tabulate(package_table, headers=[
              "ID", "Picked up", "Delivered", "Priority", "Pickup position", "Dropoff position"]))
        print("\nRobot Status:")
        print(tabulate(robot_table, headers=[
              "ID", "Idle", "Current position", "Destination position", "Current Task"]))

        time.sleep(2)  # Print updates every 2 seconds


if __name__ == "__main__":
    main()
    