import matplotlib.pyplot as plt

labels = ["Random", "Static", "Best", "DQN"]
latencies = [55.32, 49.68, 28.80, 40.26]

plt.figure()
plt.bar(labels, latencies)

plt.xlabel("Methods")
plt.ylabel("Average Latency")
plt.title("Comparison of Routing Methods")

for i, v in enumerate(latencies):
    plt.text(i, v + 1, f"{v:.2f}", ha='center')

# 🔥 مهم جدًا: احفظ قبل show
plt.savefig("results/final_results.png", dpi=300, bbox_inches='tight')

plt.show()
