import numpy as np
import matplotlib.pyplot as plt
import time


# Create a grid (maze)
def create_maze():
    maze = np.array([
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1]
    ])
    return maze


# Visualize the maze
def display_maze(maze, player_position=None, visited=None):
    plt.imshow(maze, cmap='binary', origin='upper')
    if visited is not None:
        for pos in visited:
            plt.scatter(pos[1], pos[0], color='blue', s=50)  # Mark visited cells
    if player_position:
        # Add the player (let's use a red circle for the player)
        plt.scatter(player_position[1], player_position[0], color='red', s=150)
    plt.xticks([]), plt.yticks([])  # Remove axis ticks
    plt.draw()


# Check if a position is valid
def is_valid_move(x, y, maze, visited):
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] == 0 and (x, y) not in visited


# Recursive function to navigate the maze
def explore_maze(x, y, maze, visited, path, start_position):
    # Add the current position to the visited set and path
    visited.add((x, y))
    path.append((x, y))

    # Visualization
    display_maze(maze, player_position=(x, y), visited=visited)
    plt.pause(0.5)

    # Try all possible moves
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
        nx, ny = x + dx, y + dy
        if is_valid_move(nx, ny, maze, visited):
            explore_maze(nx, ny, maze, visited, path, start_position)
            return  # Return after completing valid moves from this cell

    # If hitting an obstacle or dead-end, restart from the beginning
    print(f"Hit obstacle or dead-end at ({x}, {y}). Restarting from start...")
    time.sleep(1)  # Pause to show restart
    display_maze(maze, player_position=start_position, visited=visited)
    explore_maze(start_position[0], start_position[1], maze, visited, path, start_position)


# Main function
def run_game():
    maze = create_maze()
    start_position = (0, 1)  # Starting point of the player
    visited = set()  # Set to keep track of visited cells
    path = []  # Store the path taken

    fig, ax = plt.subplots()
    plt.ion()  # Turn on interactive mode
    display_maze(maze, player_position=start_position)

    # Start exploring the maze
    explore_maze(start_position[0], start_position[1], maze, visited, path, start_position)

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Run the game
run_game()
