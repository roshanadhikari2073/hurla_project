import os
import json
import numpy as np
from collections import deque
import random
from collections import defaultdict

# Paths for persisting learned policy and last used threshold
Q_TABLE_FILE = "logs/q_table.json"
THRESHOLD_FILE = "logs/last_threshold.txt"

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))  # Maps state -> action values
        self.replay = deque(maxlen=50)  # Memory buffer to store past experiences for replay
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor for future reward
        self.epsilon = epsilon  # Exploration rate

        # Load existing Q-table if available
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'r') as f:
                data = json.load(f)
                for state_str, q_vals in data.items():
                    self.q_table[state_str] = np.array(q_vals)

    def choose_action(self, state):
        # Epsilon-greedy strategy: explore or exploit
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        state_str = self._state_to_str(state)
        return self.actions[np.argmax(self.q_table[state_str])]

    def update(self, state, action, reward, next_state):
        # Main Q-learning update rule
        state_str = self._state_to_str(state)
        next_state_str = self._state_to_str(next_state)
        action_index = self.actions.index(action)
        best_next = np.max(self.q_table[next_state_str])
        current_q = self.q_table[state_str][action_index]
        self.q_table[state_str][action_index] = current_q + self.alpha * (
            reward + self.gamma * best_next - current_q
        )

        # Add this experience to replay memory for optional reuse
        self.replay.append((state, action, reward, next_state))

        # Save updated Q-table to disk
        self._save_q_table()

    def replay_sample(self):
        # Sample one past experience and perform a replayed Q-update
        if not self.replay:
            return
        state, action, reward, next_state = random.choice(list(self.replay))
        state_str = self._state_to_str(state)
        next_state_str = self._state_to_str(next_state)
        action_index = self.actions.index(action)
        best_next = np.max(self.q_table[next_state_str])
        current_q = self.q_table[state_str][action_index]
        self.q_table[state_str][action_index] = current_q + self.alpha * (
            reward + self.gamma * best_next - current_q
        )

    def _state_to_str(self, state):
        # Convert float metrics to a simple string bucket: e.g., "9-10-9"
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

    def _get_state(self, precision, recall, f1):
        # Discretize metrics into bins for state representation
        p_bin = int(precision * 10)
        r_bin = int(recall * 10)
        f_bin = int(f1 * 10)
        return (p_bin, r_bin, f_bin)