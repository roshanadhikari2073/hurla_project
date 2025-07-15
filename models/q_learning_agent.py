# models/q_learning_agent.py
# Double-Q Learning agent for threshold tuning. (unchanged except for _get_state)

import os
import json
import random
from collections import defaultdict, deque

import numpy as np

Q1_FILE        = "logs/q1_table.json"
Q2_FILE        = "logs/q2_table.json"
THRESHOLD_FILE = "logs/last_threshold.txt"

class QLearningAgent:
    def __init__(self, actions,
                 alpha=0.1, gamma=0.9,
                 epsilon=1.0, min_epsilon=0.05, decay=0.995,
                 replay_size=100,
                 q_cap=(-10.0, 10.0)):
        self.actions     = actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.min_epsilon = min_epsilon
        self.decay       = decay

        self.replay      = deque(maxlen=replay_size)
        self.q_min, self.q_max = q_cap

        # two separate tables for Double-Q
        self.q1_table = defaultdict(lambda: np.zeros(len(actions)))
        self.q2_table = defaultdict(lambda: np.zeros(len(actions)))

        # warm-start if available
        if os.path.exists(Q1_FILE):
            with open(Q1_FILE) as f:
                for s, vals in json.load(f).items():
                    self.q1_table[s] = np.array(vals)
        if os.path.exists(Q2_FILE):
            with open(Q2_FILE) as f:
                for s, vals in json.load(f).items():
                    self.q2_table[s] = np.array(vals)

    def choose_action(self, state):
        """ε-greedy on the average of Q1 and Q2."""
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        s     = self._state_to_str(state)
        avg_q = (self.q1_table[s] + self.q2_table[s]) / 2.0
        return self.actions[int(np.argmax(avg_q))]

    def update(self, state, action, reward, next_state):
        """Perform one Double-Q update, one replay step, decay ε, persist."""
        norm_r = np.tanh(reward / max(abs(reward), 1.0))
        use_q1 = (len(self.replay) % 2 == 0)

        A = self.q1_table if use_q1 else self.q2_table
        B = self.q2_table if use_q1 else self.q1_table

        s, s_next = self._state_to_str(state), self._state_to_str(next_state)
        a_idx     = self.actions.index(action)

        # select via A, evaluate via B
        best_next = int(np.argmax(A[s_next]))
        td_target = norm_r + self.gamma * B[s_next][best_next]
        td_error  = td_target - A[s][a_idx]
        A[s][a_idx] += self.alpha * td_error
        A[s][a_idx]  = np.clip(A[s][a_idx], self.q_min, self.q_max)

        self.replay.append((state, action, norm_r, next_state))
        self._replay_transition()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self._save_tables()

    def replay_sample(self):
        """(Not used directly) preserved for API consistency."""
        pass

    def _replay_transition(self):
        """Apply one random past experience as another TD-step."""
        if not self.replay:
            return
        st, act, rew, st_next = random.choice(list(self.replay))
        use_q1 = random.choice([True, False])
        A = self.q1_table if use_q1 else self.q2_table
        B = self.q2_table if use_q1 else self.q1_table

        s, s_next = self._state_to_str(st), self._state_to_str(st_next)
        a_idx     = self.actions.index(act)

        best_next = int(np.argmax(A[s_next]))
        td_target = rew + self.gamma * B[s_next][best_next]
        td_error  = td_target - A[s][a_idx]
        A[s][a_idx] += self.alpha * td_error
        A[s][a_idx]  = np.clip(A[s][a_idx], self.q_min, self.q_max)

    def get_last_threshold(self, fallback):
        if os.path.exists(THRESHOLD_FILE):
            try:
                return float(open(THRESHOLD_FILE).read().strip())
            except:
                pass
        return fallback

    def save_current_threshold(self, val):
        os.makedirs(os.path.dirname(THRESHOLD_FILE), exist_ok=True)
        with open(THRESHOLD_FILE, "w") as f:
            f.write(f"{val:.6f}")

    @staticmethod
    def _state_to_str(state):
        """Bucket (prec, recall, f1) into 21 discrete bins each."""
        p, r, f = state
        return f"{int(p*20)}-{int(r*20)}-{int(f*20)}"

    def _get_state(self, precision, recall, f1):
        """
        Exposed hook for hurla_pipeline:
        returns the raw (precision, recall, f1) tuple.
        """
        return (precision, recall, f1)

    def _save_tables(self):
        """Persist both Q-tables to disk."""
        os.makedirs(os.path.dirname(Q1_FILE), exist_ok=True)
        with open(Q1_FILE, "w") as f:
            json.dump({s: q.tolist() for s, q in self.q1_table.items()}, f)
        with open(Q2_FILE, "w") as f:
            json.dump({s: q.tolist() for s, q in self.q2_table.items()}, f)