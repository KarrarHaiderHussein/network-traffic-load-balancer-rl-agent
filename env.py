import numpy as np
import random


class NetworkEnv:
    def __init__(self, max_steps=50):
        self.num_paths = 3
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        self.state = self._generate_state()
        return self.state

    def _generate_state(self):
        """
        State format:
        [latency_A, congestion_A, latency_B, congestion_B, latency_C, congestion_C]
        """
        state = [] 

        for _ in range(self.num_paths):
            latency = random.uniform(10, 100)        # ms
            congestion = random.uniform(0, 1)        # 0 to 1
            state.extend([latency, congestion])

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Actions:
        0 -> choose Path A
        1 -> choose Path B
        2 -> choose Path C
        """
        self.current_step += 1

        chosen_latency = self.state[action * 2]
        chosen_congestion = self.state[action * 2 + 1]

        # Reward: lower latency and lower congestion are better
        reward = -(chosen_latency + chosen_congestion * 50)

        # Generate next state
        next_state = self._generate_state()
        self.state = next_state

        done = self.current_step >= self.max_steps

        info = {
            "chosen_path": action,
            "latency": float(chosen_latency),
            "congestion": float(chosen_congestion)
        }

        return next_state, reward, done, info

    def sample_action(self):
        return random.randint(0, self.num_paths - 1)