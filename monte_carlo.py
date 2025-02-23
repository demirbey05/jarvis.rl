from collections import defaultdict
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar


class MonteCarloAgent():

    def __init__(self, env, num_episodes=1000, gamma=1.0, epsilon=0.1):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    

    def generate_episode(self, state=None, action=None):
        if state is None:
            state = self.env.reset()
        else:
            state = self.env.reset(options={"start_state": state})
        
        if action is None:
            # Choose randomly among optimal actions
            best_actions = np.where(self.policy[state["agent"]] == np.max(self.policy[state["agent"]]))[0]
            action = np.random.choice(best_actions)
        episode = []
        while True:
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
            # Choose randomly among optimal actions for the next step
            best_actions = np.where(self.policy[state["agent"]] == np.max(self.policy[state["agent"]]))[0]
            action = np.random.choice(best_actions)
        return episode

    def mc_basic(self, max_iter=10000):
        for i in tqdm(range(max_iter), desc="MC iterations"):
            for state in range(self.env.observation_space.n):
                for action in range(self.env.action_space.n):
                    total_return = 0
                    for _ in range(self.num_episodes):
                        episode = self.generate_episode(state, action)
                        total_return += sum([reward * self.gamma**i for i, (_, _, reward) in enumerate(episode)])
                    self.q[state, action] = total_return / self.num_episodes
            
            self.policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            for state in range(self.env.observation_space.n):
                best_action = np.argmax(self.q[state])
                self.policy[state, best_action] = 1.0
        

if __name__ == "__main__":
    from grid import GridWorldEnv
    env = GridWorldEnv(size=5)
    agent = MonteCarloAgent(env,num_episodes=1)
    agent.mc_basic()
    print(agent.policy)
    print(agent.q)