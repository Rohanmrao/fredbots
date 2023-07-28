import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Example matrix and initial points
matrix = np.zeros((21, 21))
points = [(5,9)]  # Example initial positions of points

path = {0:[(5,9), (5,10), (4,10), (3,10), (2,10), (2,9), (2,8), (2,7), (2,6), (2,5), (2,4),(1,4),(0,4),(0,5),(1,5),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(3,10),(4,10),(5,10),(6,10), (7,10),(7,9),(7,8), (6,8), (7,8), (7,9),(7,8), (7,9), (7,8), (8,8), (9,8), (9,9),(10,9),(11,9),(11, 10),(12,10),(12,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(13,18),(14,18)], 
        }

goal_1 = [(2,4)]

goal_2 = [(14,18)]

def update(frame):

    # Clear the previous plot and redraw the updated points
    plt.clf()
    plt.imshow(matrix, cmap='gray')  # Show the matrix as an image
    
    # Plot the points on the matrix with unique colors based on index 'i'
    for i, point in enumerate(points):
        plt.scatter(point[1], point[0], c=f'C{i}', marker='o')  # Plot the points with unique colors

    # Mark the fixed points with a different marker (e.g., 'X')
    for point in goal_1:
        plt.scatter(point[1], point[0], c='red', marker='*', s=100)
    
    # Mark the fixed points with a different marker (e.g., 'X')
    for point in goal_2:
        plt.scatter(point[1], point[0], c='red', marker='*', s=100)

    # Mark the fixed points with a different marker (e.g., 'o')
    for i, point in enumerate(points):
        plt.scatter(point[1], point[0], c=f'C{i}', marker='o')

    # Plot the paths for each point
    for key, p in path.items():
        x, y = zip(*p)
        plt.plot(y, x, color='blue', linestyle='-', linewidth=2, alpha=0.7)

    plt.title(f"Frame: {frame}")



# Create the animation
fig = plt.figure()

# invert x and y axis
plt.gca().invert_yaxis()


ani = animation.FuncAnimation(fig, update, frames=100, interval=200)
plt.show()




