import random
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

register(
    id="MazeEnv_v1",
    entry_point= "maze_env:MazeEnv",
    max_episode_steps= 100)


class MazeEnv(gym.Env):

    def __init__(self, render_mode=None):

        self.size = 5
        self.maze = np.array([
            [1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1]
        ])
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 100, shape=(2,), dtype=int),
                "target_location": spaces.Box(0, 100, shape=(2,), dtype=int)
            }
        )
        # The 4 movement options directions
        self.action_space = spaces.Discrete(4)
        self._agent_location = np.array([1, 0], dtype=int)
        self._target_location = np.array([4, 3], dtype=int)
        self._action_to_direction = {
            0: np.array([0, 1]),  # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),  # down
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target_location": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        choices = [np.array([1, 0], dtype=int),
                   np.array([0, 1], dtype=int),
                   np.array([3, 1], dtype=int),
                   np.array([2, 4], dtype=int)]
        self._agent_location = random.choice(choices)
        self._target_location = np.array([4, 3], dtype=int)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def reward(self, tl, al):
        if np.array_equal(tl, al):
            reward = -100
        else:
            reward = -10
        return reward

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # right, up, left, down
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        target_location = np.clip(self._agent_location + direction, 0, np.array([4, 6]))
        # check if it doesn't collide

        if self.maze[tuple(target_location)] == 1:
            target_location = self._agent_location

        reward = self.reward(target_location, self._agent_location)
        self._agent_location = target_location

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward += 100 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
