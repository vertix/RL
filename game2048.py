import random
import sys


import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class Game2048(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=12, shape=(4, 4))
        self.reward_range = (0, 1000)

        self._reset()

    def _reset(self):
        self.state = np.zeros([4, 4])
        # Set up the start position
        self.state[1, 1] = 1
        self.state[2, 2] = 1
        return np.array(self.state)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def _StackRow(row):
        """Takes the row and joins it from higher indexes to lower"""
        clean_row = row[row != 0]
        result = []
        i = 0
        reward = 0
        while i < clean_row.size:
            if (i == clean_row.size - 1) or clean_row[i] != clean_row[i + 1]:
                # No joining happens
                result.append(clean_row[i])
                i += 1
            else:  # Two consequtive blocks join
                result.append(clean_row[i] + 1)
                reward = 2. ** (clean_row[i] + 1)
                i += 2

        return np.array(result + [0] * (4 - len(result)), dtype=row.dtype), reward


    def _step(self, action):
        """Performs one step given selected action. Returns step reward."""
        reward = 0.
        if action == 0:  # up
            for i in range(4):
                self.state[:, i], rew = Game2048._StackRow(self.state[:, i])
                reward += rew
        elif action == 1:  # down
            for i in range(4):
                self.state[::-1, i], rew = Game2048._StackRow(self.state[::-1, i])
                reward += rew
        elif action == 2:  # left
            for i in range(4):
                self.state[i, :], rew = Game2048._StackRow(self.state[i, :])
                reward += rew
        elif action == 3:  # right
            for i in range(4):
                self.state[i, ::-1], rew = Game2048._StackRow(self.state[i, ::-1])
                reward += rew

        empty_cells = []
        for x in range(4):
            for y in range(4):
                if self.state[x, y] == 0:
                    empty_cells.append((x, y))

        done = False
        if not empty_cells:
            done = True
        else:
            cell = random.choice(empty_cells)
            self.state[cell] = random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])

        return np.array(self.state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        for i in range(4):
            if i > 0:
                sys.stdout.write('---------------\n')

            sys.stdout.write('|'.join([str(int(v)).center(3) for v in self.state[i, :]]))
            sys.stdout.write('\n')
