import numpy as np

maze = np.array([
            [1, 2, 1, 0, 0, 0, 1],
            [3, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1]
        ])
target_location = np.array([0, 1])
print(maze[0, 1])
print(np.array([0, 3])+np.array([3, 5]))
print(maze[target_location])
