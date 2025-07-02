# ----------------------------------------
# models/q_learning_agent.py (UPDATED)
# ----------------------------------------
# q_learning_agent.py

import os
import json
import numpy as np
from collections import defaultdict

Q_TABLE_FILE = "logs/q_table.json"
THRESHOLD_FILE = "logs/last_threshold.txt"

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'r') as f:
                data = json.load(f)
                for state_str, q_vals in data.items():
                    self.q_table[state_str] = np.array(q_vals)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        state_str = self._state_to_str(state)
        return self.actions[np.argmax(self.q_table[state_str])]

    def update(self, state, action, reward, next_state):
        state_str = self._state_to_str(state)
        next_state_str = self._state_to_str(next_state)

        action_index = self.actions.index(action)
        best_next = np.max(self.q_table[next_state_str])
        current_q = self.q_table[state_str][action_index]

        self.q_table[state_str][action_index] = current_q + self.alpha * (reward + self.gamma * best_next - current_q)
        self._save_q_table()

    def _state_to_str(self, state):
        return f"{int(state[0]*10)}-{int(state[1]*10)}-{int(state[2]*10)}"

    def _save_q_table(self):
        with open(Q_TABLE_FILE, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.q_table.items()}, f)

    def get_last_threshold(self, default_value):
        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE, 'r') as f:
                try:
                    return float(f.read().strip())
                except:
                    pass
        return default_value

    def save_current_threshold(self, value):
        with open(THRESHOLD_FILE, 'w') as f:
            f.write(f"{value:.6f}")