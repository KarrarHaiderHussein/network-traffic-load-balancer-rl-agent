# 🚀 Network Traffic Load Balancer using RL Agent

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![RL](https://img.shields.io/badge/Technique-DQN-orange)

---

---

## 🧩 System Architecture

<p align="center">
  <img src="images/architecture.png" width="75%">
</p>

<p align="center">
<b>Figure 1:</b> Reinforcement Learning (DQN) agent interacting with the network environment to optimize traffic routing.
</p>

---

### 🔍 Comparison with Traditional Load Balancing

<p git pull origin main --rebasealign="center">
  <img src="images/architecture1.png" width="45%">
  <img src="images/architecture2.png" width="45%">
</p>

<p align="center">
<b>Figure 2:</b> Traditional load balancing architecture and common distribution algorithms (Round Robin, Least Connections, Hashing).
</p>

---

### 🧠 How It Works

- Incoming **network traffic** enters the environment  
- The environment computes the current **state** (load, latency)  
- The **DQN agent** selects the optimal server  
- The system returns a **reward** based on latency performance  
- The agent continuously improves decisions via **feedback loop**

---

---

## 📌 Overview

This project implements a **Reinforcement Learning-based Load Balancer** that dynamically distributes network traffic across multiple servers using a **Deep Q-Network (DQN)** agent.

The goal is to minimize latency and improve system efficiency by learning optimal routing decisions over time.

---

## 🎯 Objectives

* Reduce network congestion
* Optimize server utilization
* Minimize response latency
* Learn adaptive routing policies

---

## 🧠 Methodology

The system uses **Deep Reinforcement Learning (DQN)**:

* Environment simulates network traffic
* Agent observes system state (load, latency)
* Agent selects best server (action)
* Reward based on performance (latency reduction)

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python train.py
python plot.py
```

---

## 📊 Results

### 🔹 Final Performance

![Final Results](results/final_results.png)

### 🔹 Training Output

![Terminal Output](results/terminal_output.png)

---

## 📁 Project Structure

```
network-traffic-load-balancer-rl/
│
├── env.py               # Environment simulation
├── dqn_agent.py         # RL Agent (DQN)
├── train.py             # Training script
├── evaluate.py          # Model evaluation
├── utils.py             # Helper functions
│
├── results/
│   ├── final_results.png
│   └── terminal_output.png
│
├── images/              # README images (optional)
│
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

---

## 📈 Key Features

* Deep Q-Learning implementation
* Adaptive load balancing
* Performance visualization
* Scalable design

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* Reinforcement Learning (DQN)

---

## 🔮 Future Improvements

* Use **Advanced RL (PPO / A3C)**
* Real-world network integration
* GUI dashboard for monitoring
* Multi-agent system

---

## 👨‍💻 Author

Karrar Haider 

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
