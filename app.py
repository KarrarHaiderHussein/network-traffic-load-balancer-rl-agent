import streamlit as st
from env import NetworkEnv
from dqn_agent import DQNAgent
from train import train_dqn, run_random_policy, run_static_policy, run_best_latency_policy, evaluate_dqn

st.title("Network Traffic Load Balancer")
st.write("Simple web app to run the load balancer project")

method = st.selectbox(
    "Choose a method",
    ["Random", "Static", "Best Latency", "DQN"]
)

if st.button("Run Model"):
    env = NetworkEnv(max_steps=50)

    if method == "Random":
        total_reward, avg_latency = run_random_policy(env)
        st.success("Random policy finished")
        st.write("Total Reward:", total_reward)
        st.write("Average Latency:", avg_latency)

    elif method == "Static":
        total_reward, avg_latency = run_static_policy(env, fixed_action=0)
        st.success("Static policy finished")
        st.write("Total Reward:", total_reward)
        st.write("Average Latency:", avg_latency)

    elif method == "Best Latency":
        total_reward, avg_latency = run_best_latency_policy(env)
        st.success("Best Latency policy finished")
        st.write("Total Reward:", total_reward)
        st.write("Average Latency:", avg_latency)

    elif method == "DQN":
        agent = DQNAgent(state_size=6, action_size=3)
        train_dqn(env, agent, episodes=100)
        total_reward, avg_latency = evaluate_dqn(env, agent)
        st.success("DQN finished")
        st.write("Total Reward:", total_reward)
        st.write("Average Latency:", avg_latency)