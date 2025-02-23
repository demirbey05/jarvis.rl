from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5,target:int = 7, forbidden: list = [1,6,8,12,16,17]):
        self.time = 0
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = -1
        self._target_location = self._index_to_location(target)
        self.forbidden_grids = forbidden

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Discrete(size**2)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }    

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is None:
            options = {}
        super().reset(seed=seed, options=options)
        if options.get("start_state") is not None:
            self._agent_location = self._index_to_location(options["start_state"])
        else:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # Choose the agent's location uniformly at random
        # We will sample the target's location randomly until it does not coincide with the agent's location
        if options.get("target_state") is not None:
            self._target_location = self._index_to_location(options["target_state"])
        observation = self._get_obs()
        info = self._get_info()
        self.time = 0

        return observation, info
    # Reward -1 if you go out, 0 if you are in the grid, 1 if you reach
    def step(self,action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        # because np.clip is returns different array
        # the previous location is not the same as the agent location
        is_go_out = np.all(new_location == self._agent_location)
        is_done = np.all(new_location == self._target_location)
        is_forbidden = self._location_to_index(new_location) in self.forbidden_grids

        self._agent_location = new_location
        reward = -1 if (is_go_out or is_forbidden) else 1 if is_done else 0
        observation = self._get_obs()
        info = self._get_info()
        self.time += 1

        return observation, reward, is_done, False, info
    
    def render(self):
        print("----------------------------------------")
        print(f"Time step: {self.time}")
        # Create an empty grid
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Fetch coordinates
        agent_x, agent_y = self._agent_location
        target_x, target_y = self._target_location

        # Place target and agent in the grid
        grid[self.size - 1 - agent_y][agent_x] = 'A'
        grid[self.size - 1 - target_y][target_x] = 'T'

        # Print the grid row by row
        for row in grid:
            print(" ".join(row))

    def _index_to_location(self, index):
        return np.array([index % self.size, index // self.size])
    
    def _location_to_index(self, location):
        return location[0] + location[1] * self.size

def render_grid_policy(policy, size):
    action_mapping = {
        0: "→",  # right
        1: "↑",  # up
        2: "←",  # left
        3: "↓",  # down
    }
    # Create a grid (rows) to hold the best action for each state.
    grid_render = [["." for _ in range(size)] for _ in range(size)]
    # For each state, find its best action (assumed to be one-hot encoded)
    for state, action_dist in enumerate(policy):
        # Get the best action (if multiple are optimal the policy should be modified accordingly)
        best_action = np.where(action_dist == 1)[0][0]
        # Convert state index to grid location:
        # Using the same encoding as in grid.py: [index % size, index // size]
        col = state % size
        row = state // size
        # Adjust row ordering for rendering (grid.py uses bottom row as index 0)
        render_row = size - 1 - row
        grid_render[render_row][col] = action_mapping[best_action]
    
    # Print the grid row by row
    for row in grid_render:
        print(" ".join(row))

