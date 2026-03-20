from env import NetworkEnv
from dqn_agent import DQNAgent
import numpy as np


def choose_best_latency_path(state):
    latencies = [state[0], state[2], state[4]]
    return int(np.argmin(latencies))


def run_random_policy(env):
    state = env.reset()
    done = False
    total_reward = 0
    total_latency = 0
    steps = 0

    while not done:
        action = env.sample_action()
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_latency += info["latency"]
        steps += 1
        state = next_state

    avg_latency = total_latency / steps
    return total_reward, avg_latency


def run_static_policy(env, fixed_action=0):
    state = env.reset()
    done = False
    total_reward = 0
    total_latency = 0
    steps = 0

    while not done:
        action = fixed_action
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_latency += info["latency"]
        steps += 1
        state = next_state

    avg_latency = total_latency / steps
    return total_reward, avg_latency


def run_best_latency_policy(env):
    state = env.reset()
    done = False
    total_reward = 0
    total_latency = 0
    steps = 0

    while not done:
        action = choose_best_latency_path(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_latency += info["latency"]
        steps += 1
        state = next_state

    avg_latency = total_latency / steps
    return total_reward, avg_latency


def train_dqn(env, agent, episodes=100):
    episode_rewards = []
    episode_latencies = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_latency = 0
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size=32)

            state = next_state
            total_reward += reward
            total_latency += info["latency"]
            steps += 1

        avg_latency = total_latency / steps
        episode_rewards.append(total_reward)
        episode_latencies.append(avg_latency)

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Avg Latency: {avg_latency:.2f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

    return episode_rewards, episode_latencies


def evaluate_dqn(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    total_latency = 0
    steps = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_latency += info["latency"]
        steps += 1
        state = next_state

    agent.epsilon = old_epsilon

    avg_latency = total_latency / steps
    return total_reward, avg_latency


env = NetworkEnv(max_steps=50)
agent = DQNAgent(state_size=6, action_size=3)

print("\nTraining DQN agent...\n")
train_dqn(env, agent, episodes=100)

print("\n===== FINAL COMPARISON =====")
random_reward, random_latency = run_random_policy(env)
static_reward, static_latency = run_static_policy(env, fixed_action=0)
best_reward, best_latency = run_best_latency_policy(env)
dqn_reward, dqn_latency = evaluate_dqn(env, agent)

print(f"Random Policy  -> Total Reward: {random_reward:.2f}, Avg Latency: {random_latency:.2f}")
print(f"Static Policy  -> Total Reward: {static_reward:.2f}, Avg Latency: {static_latency:.2f}")
print(f"Best Latency   -> Total Reward: {best_reward:.2f}, Avg Latency: {best_latency:.2f}")
print(f"DQN Agent      -> Total Reward: {dqn_reward:.2f}, Avg Latency: {dqn_latency:.2f}")