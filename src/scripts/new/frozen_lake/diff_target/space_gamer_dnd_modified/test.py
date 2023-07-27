import numpy as np

obstacles = np.array([[2,0],[2,1],[2,2],[2,3],[2,4],[6,6],[6,7],[6,8],[6,9],[6,10]])

def check(test,array):
    return any(np.array_equal(x, test) for x in array)

unique = set()
for _ in range(1000):
    spawn_point = np.random.randint(0, 11, size=2)
    while check(spawn_point, obstacles):
        spawn_point = np.random.randint(0, 11, size=2)
    unique.add(tuple(spawn_point))
    # print('Spawn point:', spawn_point)  

print('Number of unique spawn points:', len(unique))
print()
print('Unique spawn points:', unique)