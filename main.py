from env import NetworkEnv
from dqn_agent import DQNAgent
from train import train_dqn, run_random_policy, run_static_policy, run_best_latency_policy, evaluate_dqn


def main():
    env = NetworkEnv(max_steps=50)
    agent = DQNAgent(state_size=6, action_size=3)

    print("\nTraining DQN agent...\n")
    train_dqn(env, agent, episodes=100)

    print("\n===== FINAL COMPARISON =====")
    random_reward, random_latency = run_random_policy(env)
    static_reward, static_latency = run_static_policy(env, fixed_action=0)
    best_reward, best_latency = run_best_latency_policy(env)
    dqn_reward, dqn_latency = evaluate_dqn(env, agent)

    print(f"Random Policy -> Total Reward: {random_reward:.2f}, Avg Latency: {random_latency:.2f}")
    print(f"Static Policy -> Total Reward: {static_reward:.2f}, Avg Latency: {static_latency:.2f}")
    print(f"Best Latency -> Total Reward: {best_reward:.2f}, Avg Latency: {best_latency:.2f}")
    print(f"DQN Agent -> Total Reward: {dqn_reward:.2f}, Avg Latency: {dqn_latency:.2f}")


if __name__ == "__main__":
    main()