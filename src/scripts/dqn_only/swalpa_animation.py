import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Example matrix and initial points
matrix = np.zeros((10, 10))
points = [(2, 3), (5, 6), (8, 1)]  # Example initial positions of points
point_paths = {frozenset([point]): [] for point in points}  # Dictionary to store paths for each point

def update(frame):
    # Example: Randomly move the points for illustration
    print("point_paths: ", point_paths)
    for i in range(len(points)):
        print("len: ", len(points))
        print("[ENTERED]")
        row, col = points[i]
        new_row = row + np.random.randint(-1, 2)
        new_col = col + np.random.randint(-1, 2)

        # Make sure the new position is within the matrix boundaries
        new_row = np.clip(new_row, 0, matrix.shape[0] - 1)
        new_col = np.clip(new_col, 0, matrix.shape[1] - 1)

        # Store the current position in the point's path
        print("points[i]: ", points[i])
        point_paths[frozenset([points[i]])].append((new_row, new_col))

        points[i] = (new_row, new_col)

    # Clear the previous plot and redraw the updated points and paths
    plt.clf()
    plt.imshow(matrix, cmap='gray')  # Show the matrix as an image

    # Plot the paths for each point in different colors
    for i, point_set in enumerate(point_paths.keys()):
        path = point_paths
        point = list(point_set)[0]  #[point_set] Convert frozenset back to tuple
        plt.plot(*zip(*path), color=f'C{i}', linestyle='-', linewidth=2)  # Different color for each path
        plt.scatter(point[1], point[0], c=f'C{i}', marker='o')  # Plot the points on the matrix

    plt.title(f"Frame: {frame}")

# Create the animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update, frames=100, interval=200)
plt.show()

# create another animation 