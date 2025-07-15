
# CICIDS2018 Threshold Adaptation with Tabular Q-Learning

This project investigates why a tabular Q-learning agent struggled to adapt threshold values effectively for anomaly detection in the CICIDS2018 dataset. Despite employing reinforcement learning and dynamic threshold adjustment techniques the agent consistently failed to achieve precision-first anomaly tuning. This document outlines the agent’s design logic, the dataset dynamics, key experimental behaviors, and the underlying reasons why tabular Q-learning turned out to be a poor fit for such a complex environment.

---

## Tools and Techniques Used

- **Dataset**: CICIDS2018 daily flow-based CSVs from February and March 2018  
- **Reinforcement Learning Strategy**: Tabular Q-Learning using ε-greedy exploration (ε ≈ 0.2)  
- **State Encoding**: Discrete buckets based on precision recall and F1 score resulting in a 3D space of 11 × 11 × 11 = 1,331 total states  
- **Reward Function**:  

Reward = (2 × TP₂) − (20 × FP₂) + small recall bonus

- **Threshold Initialization**: Derived from exponentially smoothed moving average based on training-time anchor values  
- **Threshold Actions Available**: Increase by 10%, decrease by 10% or hold  
- **Learning Update Rule**: Single-step temporal difference (TD) update  
- **Persistence**: Q-table and threshold values were saved after each batch for continuity  
- **Feature Drift Analysis**: Performed using Kolmogorov–Smirnov tests to quantify daily distribution shifts

---

## Dataset Overview

Each CSV file represented a single day's worth of flow records from CICIDS2018. We focused on over ten daily files across February and March 2018. Every batch had more than a million records and the ratio of attack to benign flows fluctuated heavily.

| Date         | Total Flows | Benign       | Attacks     |
|--------------|-------------|--------------|-------------|
| 02-14-2018   | 1,048,575   | 667,626      | 380,949     |
| 02-15-2018   | 1,048,575   | 996,077      | 52,498      |
| 02-16-2018   | 1,048,575   | 1,006,439    | 42,136      |
| 02-20–03-02  | ~1,000,000  | varies       | varies      |

### Dataset Characteristics

- The volume was consistently high with over a million rows per batch which made per-batch tuning computationally expensive  
- Class imbalance varied sharply with some days having attack prevalence as low as 0.3% and others exceeding 30%  
- Feature drift was extremely prominent with statistical tests showing that nearly all 77 features had significant distributional changes across days. Key metrics like `Flow Duration` and `Avg Packet Size` changed dramatically

---

## Q-Learning Agent Workflow

The tabular agent was intended to adapt the anomaly detection threshold across each batch using simple performance feedback.

### Step-by-Step Procedure

1. **Observe**: Compute precision recall and F1 score using the current threshold  
2. **Select Action**: Choose among three possible actions using ε-greedy logic  
 - Lower the threshold by 10%  
 - Keep it unchanged  
 - Raise it by 10%  
3. **Reward Calculation**:  

Reward = (2 × TP₂) − (20 × FP₂) + recall bonus

4. **Update the Q-table** using the temporal difference update rule:  

Q(s, a) = Q(s, a) + α × [R + γ × max(Q(s’, a’)) − Q(s, a)]

5. **Persist State**: Save the current Q-table and the latest threshold value for use with the next batch

### State Design

- The agent's state space consisted of 3D buckets based on precision recall and F1 scores  
- Each of the three metrics ranged from 0.0 to 1.0 with increments of 0.1  
- This yielded 11 × 11 × 11 = 1,331 total states in the Q-table

---

## Observed Results and Behavior

Even after testing multiple code variants changes in reward weightings and batch-level preprocessing the agent never learned a meaningful or robust thresholding policy.

### Sample Performance Across Batches

| File          | Init Thr | Final Thr   | Precision (1st→2nd) | Recall (1st→2nd) |
|---------------|----------|-------------|----------------------|------------------|
| 02-14-2018    | 8.00     | 8.00 → 8.53 | 0.918 → 0.948        | 0.703 → 0.750    |
| 02-15-2018    | 8.00     | 8.00        | 0.205 → 0.240        | 0.998 → 0.928    |
| 02-16-2018    | 8.00     | ~9.00       | 0.273 → 0.288        | 0.003 → 0.003    |
| 02-20-2018    | 8.00     | 8.00        | 0.146 → 0.143        | 0.503 → 0.503    |
| 02-21-2018    | 8.00     | ~7.20       | 1.000 → 1.000        | 0.241 → 0.241    |

### Behavior Trends

- Threshold values barely changed across batches staying close to the initial value of 8.0  
- Precision improvements were limited even when attack frequency changed  
- On days with very rare attacks recall often collapsed and the agent failed to recover  
- The agent showed inertia and rarely escaped its initial behavior pattern

---

## Root Causes of Failure

### Simplistic State Encoding

By collapsing a batch of over a million rows into just three numbers the agent was deprived of deeper information like actual attack density class distribution shift or anchor drift

### Non-Stationary Input Conditions

Every day had a completely different data profile. Standard Q-learning assumes a stable Markov process but that assumption breaks in dynamic network data like this

### Weak and Sparse Rewards

With attack counts below 1% reward values were extremely small giving the agent very little signal to learn from

### Insufficient Exploration and Learning

With only one episode per batch and a low exploration rate the agent had very few chances to meaningfully try new actions

### Misaligned Reward Surface

The use of a steep false positive penalty made the agent overly cautious. It defaulted to safe non-adaptive thresholds instead of making learning-driven changes

---

## Why Tabular Q-Learning Did Not Fit

Tabular Q-learning works best when the environment is:

- Small and discrete  
- Statistically stable over time  
- Provides frequent feedback  
- Allows repeated exploration of every state-action combination

Our threshold adaptation problem violated every one of those expectations. The environment was large noisy and shifting with feedback too sparse to train a simple lookup-table-based agent

---

## Conclusion and Next Steps

Tabular Q-learning was not capable of handling the demands of this setting. Its shallow state representation rigid update scheme and lack of adaptability made it inadequate. The system was not able to evolve or converge under such high variability and reward sparsity

### What Comes Next

To address these limitations we plan to implement more advanced agents:

- **Deep Q-Networks** with richer input features and neural function approximation  
- **Policy Gradient Models** that can learn stochastic policies  
- **Replay Buffers and Target Networks** to provide stable and long-term learning  
- **Expanded State Descriptions** that include anchor stats class ratios and recent batch trends  
- **Smoothed and Normalized Rewards** that make learning gradients more consistent

With these enhancements we expect the system to be more intelligent and flexible and better able to handle real-world traffic conditions while achieving high precision anomaly detection

---