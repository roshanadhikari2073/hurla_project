# CICIDS2018 Threshold Adaptation with Tabular Q-Learning

This project explores the limitations of applying a tabular Q-learning agent to adaptive threshold tuning in the CICIDS2018 dataset. Despite several reinforcement learning strategies and threshold adjustment techniques, the agent consistently failed to converge on effective precision-first anomaly detection. This document presents a comprehensive breakdown of the agent’s design, data dynamics, observed behaviors, and why tabular Q-learning is fundamentally inadequate for this type of non-stationary, high-volume environment.

---

## Tools and Techniques

- **Dataset**: CICIDS2018 daily flow-based CSVs (Feb–Mar 2018)
- **RL Method**: Tabular Q-Learning with ε-greedy exploration (ε ≈ 0.2)
- **State Representation**: Discretized buckets of (precision, recall, F1), 11×11×11 = 1,331 states
- **Reward Formula**: `2·TP₂ − 20·FP₂ + small recall bonus`
- **Threshold Initialization**: EMA-smoothed anchor from training data
- **Threshold Adjustment Actions**: ±10% or hold
- **Update Rule**: One-step temporal difference (TD)
- **Persistence**: Q-table and threshold saved per batch for continuity
- **Feature Drift Assessment**: Kolmogorov–Smirnov (KS) statistical test

---

## Dataset Overview

Each batch represented a day’s worth of flow records in CICIDS2018. We used over 10 daily files between February and March 2018, with more than 1 million flows per batch. Attack-to-benign ratios varied significantly across days, and feature distributions shifted widely.

| Date         | Total Flows | Benign       | Attacks     |
|--------------|-------------|--------------|-------------|
| 02-14-2018   | 1,048,575   | 667,626      | 380,949     |
| 02-15-2018   | 1,048,575   | 996,077      | 52,498      |
| 02-16-2018   | 1,048,575   | 1,006,439    | 42,136      |
| 02-20–03-02  | ~1,000,000  | Varies       | Varies      |

### Key Characteristics

- **High volume**: Over 1 million samples per day, making batch adaptation computationally demanding.
- **Severe class imbalance**: Attack prevalence ranged from approximately 36% to below 1%.
- **Drifted features**: KS-tests revealed statistically significant distributional shifts across all 77 features. Metrics like `Flow Duration`, `Total Length of Fwd Packets`, and `Avg Packet Size` showed magnitude-level variability between days.

---

## Q-Learning Agent Workflow

The agent was designed to adapt the anomaly-score threshold across batches based on per-batch evaluation feedback.

### Step-by-Step Process

1. **Observe**: Apply current threshold to compute precision, recall, and F1-score.
2. **Choose Action**: ε-greedy selection among [−10%, 0%, +10%] threshold adjustment.
3. **Reward Calculation**:
   \[
   \text{Reward} = 2·TP₂ - 20·FP₂ + \text{(recall bonus)}
   \]
4. **Update Q-Table**:
   \[
   Q(s, a) \leftarrow Q(s, a) + \alpha \left[R + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\right]
   \]
5. **Persist**: Save updated Q-table and new threshold for the next batch.

### State Space

- Discrete 3D grid of (precision, recall, F1) values
- Each dimension bucketed from 0.0 to 1.0 in 0.1 increments
- Total number of states: 11 × 11 × 11 = 1,331

---

## Observations and Outcomes

Despite dozens of runs with adjusted hyperparameters, different reward functions, and batch-specific preprocessing enhancements, the agent failed to learn meaningful threshold policies.

### Representative Results

| File          | Init Thr | Final Thr   | Precision (1st→2nd) | Recall (1st→2nd) |
|---------------|----------|-------------|----------------------|------------------|
| 02-14-2018    | 8.00     | 8.00 → 8.53 | 0.918 → 0.948        | 0.703 → 0.750    |
| 02-15-2018    | 8.00     | 8.00        | 0.205 → 0.240        | 0.998 → 0.928    |
| 02-16-2018    | 8.00     | ~9.00       | 0.273 → 0.288        | 0.003 → 0.003    |
| 02-20-2018    | 8.00     | 8.00        | 0.146 → 0.143        | 0.503 → 0.503    |
| 02-21-2018    | 8.00     | ~7.20       | 1.000 → 1.000        | 0.241 → 0.241    |

### Failure Patterns

- **Minimal threshold change**: Most thresholds remained pinned around the initial value of 8.0.
- **Precision plateau**: In low-attack scenarios, precision failed to improve significantly.
- **Recall collapse**: On sparse-attack days, the agent was unable to compensate by lowering the threshold.
- **Inertia**: Thresholds stagnated despite reward shaping, EMA smoothing, and dynamic anchor capping.

---

## Root Causes of Failure

### 1. Over-Simplified State Encoding

The agent represented each batch by only three scalar values—precision, recall, and F1—eliminating important contextual data such as class balance, anchor magnitude, or feature drift. This severely limited the agent’s capacity to generalize or differentiate batch complexity.

### 2. Non-Stationary, Noisy Input Distribution

Each day’s batch represented a radically different distribution due to network context changes. Traditional Q-learning assumes a Markov process with stationary transitions, which was clearly violated in this scenario.

### 3. Sparse and Low-Magnitude Rewards

When attack prevalence dropped below 1%, both true positives and false positives were rare, resulting in near-zero rewards. The agent had little signal to differentiate between good and bad actions.

### 4. Inadequate Exploration and Learning Depth

With ε=0.2 and a single episode per batch, the agent explored very little. Its one-step TD updates barely influenced the Q-table, which lacked the granularity to adapt across highly variable days.

### 5. Poor Reward Surface Design

The reward’s steep FP penalty (−20 per false positive) made the agent risk-averse. It gravitated toward inaction and preserved status quo behaviors, failing to discover thresholds that truly improved precision-recall trade-offs.

---

## Why Tabular Q-Learning Was a Poor Fit

Tabular Q-learning is ideal for environments that are:

- Small and finite in state space
- Stationary in transitions and rewards
- Frequent in reward delivery (dense feedback)
- Trained over many episodes with full state-action coverage

The CICIDS2018 batches violated every one of these assumptions. The combination of high-dimensionality, concept drift, and rare-event detection made it an ill-suited environment for traditional tabular methods.

---

## Conclusion and Future Directions

The tabular Q-learning agent did not perform adequately under the demands of the CICIDS2018 setting. Its simplified state space, sparse rewards, and inability to adjust to temporal shifts rendered it ineffective for precision-oriented threshold tuning. The system remained mostly static and blind to the contextual dynamics of each batch.

### Planned Enhancements

To overcome these limitations, the following improvements are planned:

- **Deep Q-Networks (DQN)** to leverage neural approximations over continuous state inputs
- **Policy Gradient Methods** to enable smooth, stochastic action selection
- **Replay Memory and Target Networks** for more stable Q-value propagation
- **Context-Rich States** including anchor trends, class distributions, and recent batch metadata
- **Reward Normalization** to ensure meaningful, scale-aware feedback for precision-first learning

These enhancements aim to produce a robust, intelligent, and adaptive anomaly detection framework suitable for real-world network environments.

---

## Contact

For technical inquiries, implementation details, or research discussions, please open an issue in this repository or reach out through the GitHub discussion board.
