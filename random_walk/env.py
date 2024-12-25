from gymnasium.spaces import Discrete

class RandomWalkEnv:
    def __init__(self):
        self.action_space = [-1,1] # -1 -> Left, 1 -> Right
        self.state = 3
        self.state_space = Discrete(7)

    def reset(self):
        self.state = 3
        return self.state

    def step(self, action):
        self.state += action
        if self.state == 6:
            reward = 1
        else:
            reward = 0
        done = False
        if self.state == 0 or self.state == 6:
            done = True
        return self.state, reward, done