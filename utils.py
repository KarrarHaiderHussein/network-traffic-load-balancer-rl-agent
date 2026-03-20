import numpy as np

def choose_best_latency_path(state):
    return int(np.argmin(state))

def compute_average(values):
    return sum(values) / len(values) if len(values) > 0 else 0