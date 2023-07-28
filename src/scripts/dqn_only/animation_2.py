import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Example matrix and initial points
matrix = np.zeros((21, 21))
points = [(5,9),(19,11),(9,11),(11,11)]  # Example initial positions of points

path = {0:[(5,9), (5,10), (4,10), (3,10), (2,10), (2,9), (2,8), (2,7), (2,6), (2,5), (2,4),(1,4),(0,4),(0,5),(1,5),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(3,10),(4,10),(5,10),(6,10), (7,10),(7,9),(7,8), (6,8), (7,8), (7,9),(7,8), (7,9), (7,8), (8,8), (9,8), (9,9),(10,9),(11,9),(11, 10),(12,10),(12,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(13,18),(14,18)], 
        1: [(19,11),(19,12),(19,13),(18,13),(17,13),(17,14),(17,15),(17,16),(16,16),(15,16),(14,16),(14,15),(14,14),(14,13),(14,12),(14,11),(14,10),(13,10),(12,10),(11,10),(11,9),(10,9),(9,9),(9,8),(8,8),(9,8),(8,8),(8,7),(7,7),(6,7),(6,8),(6,9),(5,9),(5,10),(5,11),(5,12),(5,13),(5,14),(4,14),(3,14),(3,15),(3,16),(3,17),(3,18)],
        2: [(9,11), (9,10),(9,9),(9,8),(8,8),(7,8),(6,8),(6,9),(6,10),(5,10),(5,11),(5,12),(5,13),(4,13),(3,13),(3,14),(2,14),(2,15),(1,15),(0,15),(0,16),(1,16),(2,16),(3,16),(4,16),(5,16),(5,15),(5,14),(5,13),(6,13),(6,12),(6,11),(6,10),(7,10),(7,9),(7,8),(8,8),(9,8),(9,9),(9,10),(9,11),(10,11),(11,11),(11,12),(12,12),(13,12),(14,12),(15,12)],
        3: [(11,11),(11,10),(11,9),(10,9),(9,9),(9,8),(9,7),(8,7),(8,8),(9,8),(10,8),(11,8),(11,9),(11,10),(12,10),(12,11),(12,12),(13,12),(13,13),(13,14),(13,15),(13,16),(13,17)]}


obs = []

# red
# orange
# green 

goal_1 = [(2,4),(17,16),(8,7)]

goal_2 = [(14,18),(3,18),(15,12)]

def update(frame):

    # Clear the previous plot and redraw the updated points
    plt.clf()
    plt.imshow(matrix, cmap='gray')  # Show the matrix as an image
    
    # Plot the points on the matrix with unique colors based on index 'i'
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], c=f'C{i}', marker='o')  # Plot the points with unique colors

    # Mark the fixed points with a different marker (e.g., 'X')
    for point in goal_1:
        plt.scatter(point[0], point[1], c='red', marker='*', s=100)
    
    # Mark the fixed points with a different marker (e.g., 'X')
    for point in goal_2:
        plt.scatter(point[0], point[1], c='red', marker='*', s=100)

    # Mark the fixed points with a different marker (e.g., 'o')
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], c=f'C{i}', marker='o')

    # Plot the paths for each point
    for key, p in path.items():
        x, y = zip(*p)
        plt.plot(x, y, c=f'C{key}', linestyle='-', linewidth=2, alpha=0.7)

    plt.title(f"Frame: {frame}")



# Create the animation
fig = plt.figure()

# flip y axis to match matrix





# ani = animation.FuncAnimation(fig, update, frames=100, interval=200)
update(frame=0)
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()





