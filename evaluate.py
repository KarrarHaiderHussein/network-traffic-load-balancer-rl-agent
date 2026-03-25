from env import NetworkEnv
from dqn_agent import DQNAgent
from utils import choose_best_latency_path

def evaluate_agent():
    env = NetworkEnv()
    agent = DQNAgent(env.state_size, env.action_size)

    state = env.reset()
    done = False

    total_latency = 0
    steps = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        total_latency += info["latency"]
        state = next_state
        steps += 1

    avg_latency = total_latency / steps
    print(f"Evaluation Avg Latency: {avg_latency:.2f}")

if __name__ == "__main__":
    evaluate_agent()