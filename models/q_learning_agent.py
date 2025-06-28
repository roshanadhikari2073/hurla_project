# ----------------------------------------
# models/q_learning_agent.py (UPDATED)
# ----------------------------------------
# We add save/load functions to persist the Q-table.

import numpy as np
import os

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.action_space = [0, 1, 2]  # DECREASE, KEEP, INCREASE

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx):
        best_next = np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action] += self.lr * (reward + self.gamma * best_next - self.q_table[state_idx, action])

    def save_q_table(self, path):
        np.save(path, self.q_table)

    def load_q_table(self, path):
        if os.path.exists(path):
            self.q_table = np.load(path)
            print(f"Loaded Q-table from {path}")
        else:
            print("Q-table file not found. Using fresh Q-table.")