class WindyEnv:
    def __init__(self, n, m, wind):
        self.n = n
        self.m = m
        self.wind = wind
        self.state_space = [(i, j) for i in range(n) for j in range(m)]
        self.action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.target = (7, 3)

    def reset(self):
        self.state = (0, 3)
        return self.state
    
    def step(self, action):
        x, y = self.state
        dx, dy = action
        x += dx
        y += dy
        x = max(0, min(self.n - 1, x))
        y += self.wind[x]
        y = max(0, min(self.m - 1, y))
        self.state = (x, y)
        if self.state == self.target:
            return self.state, 0, True
        return self.state, -1, False
    
