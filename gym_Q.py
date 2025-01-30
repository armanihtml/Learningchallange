from collections import defaultdict
import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from maze_env import MazeEnv
from maze_agent import MazeAgent
import numpy as np
import matplotlib.pyplot as plt
import time


def create_maze():
    maze = np.array([
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1]
    ])
    return maze


def display_maze(maze, player_position=None, visited=None):
    plt.clf()  # Clear the previous frame
    plt.imshow(maze, cmap='binary', origin='upper')

    if visited is not None:
        for pos in visited:
            plt.scatter(pos[0], pos[1], color='blue', s=50)  # Mark visited cells

    if player_position:
        # Add the player (red circle for the player)
        # print(player_position[0], player_position[1])
        plt.scatter(player_position[0], player_position[1], color='red', s=150)

    plt.xticks([]), plt.yticks([])  # Remove axis ticks
    # plt.pause(0.1)  # Small pause to allow real-time updates
    plt.draw()


def is_valid_move(x, y, maze, visited):
    return 0 <= x < maze.shape[1] and 0 <= y < maze.shape[0] and maze[y, x] == 0


def explore_maze(x, y, maze, visited, path, start_position, agent1 = 0, obs1 = 0):
    # Add the current position to the visited set and path
    visited.add((x, y))
    path.append((x, y))

    # Visualization
    display_maze(maze, player_position=(x, y), visited=visited)
    plt.pause(0.5)
    if x == 3 and y == 4:
        print("Hit the goal")
        return
    if len(path) == 500:
        return
    # Try all possible moves
    if agent == 0:
        my_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random_item = random.choice(my_list)
        dy = random_item[0]
        dx = random_item[1]
        nx, ny = x + dx, y + dy
        next_obs = 0
    else:   # here choosing the option
        action = agent1.get_action(obs1)
        real_action = agent1.action_meaning(action)
        # print(real_action)
        dy = int(real_action[0])
        dx = int(real_action[1])
        nx, ny = x + dx, y + dy
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = tuple(next_obs.values())
        next_obs = (next_obs[0][0], next_obs[0][1], next_obs[1][0], next_obs[1][1])
        print(next_obs)
        print("Q-values for obs:", agent1.q_values[next_obs])

    if is_valid_move(nx, ny, maze, visited):
        explore_maze(nx, ny, maze, visited, path, start_position, agent1, next_obs)
        return
    else:
        print("BONK")
        explore_maze(x, y, maze, visited, path, start_position, agent1, next_obs)
        return  # Return after completing valid moves from this cell


def run_game(agent1 = 0, obs1 = 0, start_pos = 0):
    maze = create_maze()
    start_position = start_pos  # Starting point of the player
    visited = set()  # Set to keep track of visited cells
    path = []  # Store the path taken

    fig, ax = plt.subplots()
    plt.ion()  # Turn on interactive mode
    display_maze(maze, player_position=start_position)

    # Start exploring the maze
    explore_maze(start_position[0], start_position[1], maze, visited, path, start_position, agent1, obs1)

    plt.ioff()  # Turn off interactive mode
    plt.show()
    print(path)


learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("maze_env:MazeEnv_v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = MazeAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    obs = tuple(obs.values())
    obs = (obs[0][0], obs[0][1], obs[1][0], obs[1][1])
    done = False
    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = tuple(next_obs.values())
        next_obs = (next_obs[0][0], next_obs[0][1], next_obs[1][0], next_obs[1][1])
        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


# env = gym.make("Blackjack-v1", sab=False, render_mode="human")
#
# obs, info = env.reset()
# done = False
# while not done:
#     action = agent.get_action(obs)
#     next_obs, reward, terminated, truncated, info = env.step(action)
#
#     # update the agent
#     agent.update(obs, action, reward, terminated, next_obs)
#
#     # update if the environment is done and the current obs
#     done = terminated or truncated
#     obs = next_obs
for i in range(10):

    obs, info = env.reset()

# TODO: Chance Obs
# TODO: Take a look at the reward structure
    obs = tuple(obs.values())
    obs = (obs[0][0], obs[0][1], obs[1][0], obs[1][1])
    start_y = obs[0]
    start_x = obs[1]
    start_game = (start_x, start_y)

# my_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# random_item = random.choice(my_list)
# print("Hello")
# print(random_item)
    run_game(agent, obs, start_game)
