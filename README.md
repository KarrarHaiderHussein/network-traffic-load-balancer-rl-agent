# Network Traffic Load Balancer RL Agent

## Project Overview

This project implements a Reinforcement Learning-based network traffic load balancer using a Deep Q-Network (DQN) agent. The objective is to dynamically select the best network path to minimize latency and reduce congestion.

## Objective

The main goal of this project is to train an intelligent agent capable of selecting the optimal network path based on real-time network conditions such as latency and congestion.

## Environment Description

A custom simulation environment was developed using Python. The environment consists of three network paths, each characterized by:

* Latency
* Congestion

The system state is represented as:

[latency_A, congestion_A, latency_B, congestion_B, latency_C, congestion_C]

## Actions

The agent can choose one of the following actions:

* 0 → Select Path A
* 1 → Select Path B
* 2 → Select Path C

## Reward Function

The reward function is designed to penalize high latency and congestion. This encourages the agent to select faster and less congested paths.

## Implemented Methods

The following routing strategies were implemented and compared:

* Random Policy
* Static Routing Policy
* Best Latency Heuristic
* Deep Q-Network (DQN)

## Project Structure

network-traffic-load-balancer-rl/
│
├── env.py
├── dqn_agent.py
├── train.py
├── plot.py
├── README.md
├── requirements.txt
├── .gitignore
└── results/
├── comparison.txt
├── final_results.png
└── terminal_output.png

## Files Description

* `env.py`: Defines the custom network simulation environment.
* `dqn_agent.py`: Implements the Deep Q-Network (DQN) model.
* `train.py`: Handles training and evaluation of all routing methods.
* `plot.py`: Generates and saves the performance comparison graph.

## Results

Final average latency values:

* Random Policy: 55.32
* Static Policy: 49.68
* Best Latency Heuristic: 28.80
* DQN Agent: 40.26

## Results Visualization

![Routing Comparison](results/final_results.png)

The results show that the DQN agent significantly improves performance compared to random and static routing approaches. The best-latency heuristic achieves the lowest latency in this simplified environment due to direct selection of the minimum latency path.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run training:

python train.py

Generate the comparison plot:

python plot.py

## Output

All results and generated files are stored inside the `results/` directory.

## Conclusion

This project demonstrates the effectiveness of Reinforcement Learning in solving network traffic routing problems. The DQN agent learned to make better decisions compared to traditional routing strategies in a dynamic environment.
