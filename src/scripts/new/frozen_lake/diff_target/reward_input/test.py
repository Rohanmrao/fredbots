import numpy as np

b = np.array([[2,3], [4,5]])
a = np.array((0,3))

print(a + np.array([1, 0]) in b)