from collections import defaultdict
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar


class MonteCarloAgent():

    def __init__(self, env, num_episodes=1000, gamma=0.9):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = np.zeros((env.observation_space.n, env.action_space.n))
        for state in range(env.observation_space.n):
            random_action = np.random.choice(env.action_space.n)
            self.policy[state, random_action] = 1.0
    

    def generate_episode(self, state=None, action=None,episode_length=30):
        if state is None:
            state = self.env.reset()
        else:
            state = self.env.reset(options={"start_state": state})
        
        if action is None:
            # Choose randomly among optimal actions
            state_index = self.env._location_to_index(state["agent"])
            best_actions = np.where(self.policy[state_index] == np.max(self.policy[state_index]))[0]
            action = np.random.choice(best_actions)
        episode = []
        i = 0
        while i < episode_length:
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
            # Choose randomly among optimal actions for the next step
            state_index = self.env._location_to_index(state["agent"])
            best_actions = np.where(self.policy[state_index] == np.max(self.policy[state_index]))[0]
            action = np.random.choice(best_actions)
            i += 1
        return episode

    def mc_basic(self, max_iter=30):
        for i in tqdm(range(max_iter)):
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
    agent = MonteCarloAgent(env, num_episodes=1)
    agent.mc_basic()
    print("Policy Grid:")
    env.render_policy(agent.policy)
