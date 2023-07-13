import time
from threading import Thread
import numpy as np
from tabulate import tabulate

# Class to represent a package
class Packages:
    def __init__(self, package_id, start_coords, dest_coords, priority):
        self.package_id = package_id
        self.start_coords = start_coords
        self.dest_coords = dest_coords
        self.priority = priority
        self.status = "waiting for pick up"
        self.waiting_time = 0

    def update_status(self, new_status):
        self.status = new_status

    def increment_waiting_time(self):
        self.waiting_time += 1

# Class to represent a robot
class Atoms:
    def __init__(self, robot_id, battery_capacity, current_coords):
        self.robot_id = robot_id
        self.battery_capacity = battery_capacity
        self.current_coords = current_coords
        self.package = None
        self.status = "idle"
        self.destination_coords = None

    def assign_package(self, package):
        self.package = package
        self.status = "going to pick up"
        self.destination_coords = package.start_coords

    def update_status(self, new_status):
        self.status = new_status

    def update_coords(self, new_coords):
        self.current_coords = new_coords

    def consume_battery(self):
        if self.status == "going to pick up" or self.status == "in transit":
            battery_consumption_rate = 0.1
        else:
            battery_consumption_rate = 0.05  # Idle robots lose battery at a slower rate

        self.battery_capacity -= battery_consumption_rate

# Euclidean distance between two coordinates
def calculate_distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#Function for assigning tasks to robots
def assign_tasks(robots, packages, time_threshold):
    while True:
        for robot in robots:
            if robot.status == "idle":
                closest_package = None
                min_distance = float('inf') # nothing so far
                highest_priority = float('-inf')
                critical_package_assigned = False

                for package in packages:
                    if package.status == "waiting for pick up":
                        distance = calculate_distance(robot.current_coords, package.start_coords) #euclidean distance
                        if package.priority == 1: #emergency package
                            if not critical_package_assigned or (distance < min_distance and package.priority > highest_priority):
                                robot.assign_package(package)
                                package.update_status("picked up by robot " + str(robot.robot_id))
                                robot.destination_coords = package.dest_coords
                                robot.update_status("in transit")
                                print(f"Robot {robot.robot_id} will pick up Critical Package {package.package_id}")
                                critical_package_assigned = True
                                min_distance = distance
                                highest_priority = package.priority
                        elif distance < min_distance and not critical_package_assigned: #priority and distances are written here
                            min_distance = distance
                            closest_package = package
                            highest_priority = package.priority
                
                if closest_package is not None and not critical_package_assigned:
                    robot.assign_package(closest_package)
                    closest_package.update_status("picked up by robot " + str(robot.robot_id))
                    robot.destination_coords = closest_package.dest_coords
                    robot.update_status("in transit")
                    print(f"Robot {robot.robot_id} will pick up Package {closest_package.package_id}")

        for package in packages:
            if package.status == "waiting for pick up":
                package.increment_waiting_time()

        time.sleep(1)  # Check for tasks every second

        # Check package waiting time threshold
        for package in packages:
            if package.waiting_time >= time_threshold:
                print(f"Package {package.package_id} has been waiting for {package.waiting_time} seconds. Prioritize moving it.")

# Function for robot movement
def move_robot(robot):
    while True:
        if robot.status == "going to pick up" or robot.status == "in transit":
            robot.consume_battery()

        time.sleep(1)  # Update battery every second


def main():
    # packages
    packages = [
        Packages(1, (2, 2), (8, 8), 2),
        Packages(2, (5, 5), (3, 1), 3),
        Packages(3, (1, 1), (9, 9), 1),
        Packages(4, (7, 7), (4, 6), 4)
    ]

    # robots
    robots = [
        Atoms(1, 100, (0, 0)),
        Atoms(2, 100, (10, 10)),
        Atoms(3, 100, (5, 5))
    ]

    time_threshold = 10  # Time threshold for package waiting time (in seconds)

    # Start task assignment thread
    assign_thread = Thread(target=assign_tasks, args=(robots, packages, time_threshold))
    assign_thread.daemon = True
    assign_thread.start()

    # Start robot movement threads
    robot_threads = []
    for robot in robots:
        t = Thread(target=move_robot, args=(robot,))
        t.daemon = True
        t.start()
        robot_threads.append(t)

    # package and robot updates
    while True: 
        package_table = []
        for package in packages:
            package_table.append([
                package.package_id,
                package.status,
                package.priority,
                package.start_coords,
                package.waiting_time
            ])

        robot_table = []
        for robot in robots:
            robot_table.append([
                robot.robot_id,
                robot.status,
                robot.current_coords,
                robot.destination_coords,
                robot.battery_capacity
            ])

        # Print table
        print("\n")
        print("Package Schedule:")
        print(tabulate(package_table, headers=["ID", "Status", "Priority", "Start Coords", "Waiting Time"]))
        print("\nRobot Status:")
        print(tabulate(robot_table, headers=["ID", "Status", "Current Coords", "Destination Coords", "Battery"]))

        time.sleep(2)  # Print updates every 2 seconds



main()
