import unittest
import numpy as np
from grid import GridWorldEnv

class TestGridWorldEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv(size=5)
        # Initialize np_random for reset
        self.env.np_random = np.random.default_rng(42)
    
    def test_reset(self):
        obs, info = self.env.reset()
        agent = obs["agent"]
        target = obs["target"]
        # Agent and target should be within bounds
        self.assertTrue(np.all(agent >= 0) and np.all(agent < self.env.size))
        self.assertTrue(np.all(target >= 0) and np.all(target < self.env.size))
        # Agent and target should not be equal
        self.assertFalse(np.all(agent == target))
        
    def test_step_reward_done(self):
        # Place agent and target such that taking action 0 (right) reaches the target.
        self.env._agent_location = np.array([0, 0], dtype=np.int32)
        self.env._target_location = np.array([1, 0], dtype=np.int32)
        obs, reward, is_done, _, info = self.env.step(0)
        # Expecting success: reward +1 and done flag True.
        self.assertTrue(is_done)
        self.assertEqual(reward, 1)
        np.testing.assert_array_equal(self.env._agent_location, np.array([1, 0], dtype=np.int32))
        
    def test_step_go_out_reward(self):
        # Place the agent at the right edge; moving right (action 0) should not change its position.
        self.env._agent_location = np.array([self.env.size - 1, 2], dtype=np.int32)
        self.env._target_location = np.array([0, 0], dtype=np.int32)  # dummy target
        obs, reward, is_done, _, info = self.env.step(0)
        # Agent's new location remains the same, so reward should be -1 and done flag False.
        self.assertEqual(reward, -1)
        self.assertFalse(is_done)
        np.testing.assert_array_equal(self.env._agent_location, np.array([self.env.size - 1, 2], dtype=np.int32))
        
    def test_render_output(self):
        # Set specific positions and time step
        self.env._agent_location = np.array([2, 3], dtype=np.int32)
        self.env._target_location = np.array([4, 1], dtype=np.int32)
        self.env.time = 5
        
        # Capture the output of the render function.
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()
            self.env.render()
            output = sys.stdout.getvalue()
            self.assertIn("Time step: 5", output)
            self.assertIn("Agent location:", output)
            self.assertIn("Target location:", output)
        finally:
            sys.stdout = old_stdout

if __name__ == '__main__':
    unittest.main()