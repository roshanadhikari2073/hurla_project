CICIDS2018 Q-Learning Threshold Tuning: Failure Analysis Report

Tools and Techniques Used
	•	Dataset: CICIDS2018 daily flow CSVs
	•	Reinforcement Learning: Tabular Q-Learning (ε-greedy policy)
	•	State Discretization: (Precision × Recall × F1), bucketed into 11×11×11 states
	•	Reward Function: Reward = 2·TP₂ − 20·FP₂ (+ recall bonus)
	•	Adaptation Objective: Batch-wise threshold tuning for precision-first detection
	•	Additional Techniques:
	•	EMA-smoothed thresholds with anchor initialization
	•	Per-batch zero-dim suppression
	•	Q-table persistence and forward updates
	•	Kolmogorov–Smirnov test for feature drift detection

⸻

Overview

This project attempted to apply a tabular Q-learning agent to adaptively tune anomaly-score thresholds across daily batches in the CICIDS2018 dataset. The goal was to maintain high precision while allowing minimal recall loss, even under day-to-day distribution shifts. Despite various tuning strategies and reward formulations, the agent failed to converge on effective thresholds across the sequence of real-world batches.

⸻

1. Dataset Characteristics

We processed 10+ daily CSV files from February and March 2018. These batches contained over a million flows each, exhibiting significant fluctuations in attack-to-benign ratios and high feature drift.

File	Total Flows	Benign	Attacks
02-14-2018.csv	1,048,575	667,626	380,949
02-15-2018.csv	1,048,575	996,077	52,498
02-16-2018.csv	1,048,575	1,006,439	42,136
02-20–03-02	Similar	Varies	Varies

Key Challenges:
	•	Massive daily volume strained agent responsiveness.
	•	Drastic class imbalance shifts, e.g., Feb 14 had ~36% attacks, Feb 15 dropped to <5%, with other days under 1%.
	•	Feature drift confirmed via Kolmogorov–Smirnov tests; nearly all 77 features changed statistically across batches, often by orders of magnitude (e.g., flow duration).

⸻

2. Agent Workflow

The tabular agent operated on a fixed state-action-reward-update cycle as follows:
	1.	Observe: Compute precision, recall, and F1 after applying the current threshold.
	2.	Choose: Use ε-greedy policy (ε ≈ 0.2) to decrease (−10%), hold, or increase (+10%) the threshold.
	3.	Reward: Apply shaped reward function to incentivize precision:
R = 2·TP₂ − 20·FP₂ + \text{small recall bonus}
	4.	Update: Use one-step Temporal Difference (TD) rule to update the Q-table:
Q(s, a) = Q(s, a) + \alpha [R + \gamma \max_a’ Q(s’, a’) - Q(s, a)]
	5.	Persist: Save updated Q-table and next threshold to disk for use on the following batch.

The state space was defined as 11 buckets each for precision, recall, and F1, yielding 1,331 discrete states.

⸻

3. Experimental Observations

Empirical Results (Selected Files):

File	Init Thr	Final Thr	Precision (1st→2nd)	Recall (1st→2nd)
02-14-2018.csv	8.00	8.00→8.53	0.918 → 0.948	0.703 → 0.750
02-15-2018.csv	8.00	8.00	0.205 → 0.240	0.998 → 0.928
02-16-2018.csv	8.00	~9.00	0.273 → 0.288	0.003 → 0.003
02-20-2018.csv	8.00	8.00	0.146 → 0.143	0.503 → 0.503
02-21-2018.csv	8.00	~7.20	1.000 → 1.000	0.241 → 0.241

Key Failure Patterns:
	•	Threshold inertia: Most thresholds remained close to their initial value of 8.0.
	•	Stagnant precision gains: Minor improvements (~0.02–0.05), often failing to surpass the 0.95 precision target.
	•	Recall collapse on sparse days: No significant response by the agent to salvage missed detections when attack prevalence dropped below 1%.

⸻

4. Diagnosed Root Causes

1. Over-Coarse State Encoding
	•	State represented only by three metrics (precision, recall, F1), ignoring context like anchor scores, day-specific drift, or attack ratios.

2. Non-Stationary Environment
	•	Each batch’s data distribution was significantly different. Classical Q-learning assumes a stable transition model, which is violated here.

3. Sparse and Weak Reward Signals
	•	On low-attack days, TP and FP counts were too small to yield meaningful gradients in the Q-values.

4. Limited Exploration
	•	ε=0.2 and one episode per batch led to very limited threshold testing. No long-term state-action learning emerged.

5. Skewed Reward Weights
	•	Overweighting FP penalties pushed the agent toward conservative behaviors, often avoiding adjustments that could have improved recall.

⸻

5. Why Tabular Q-Learning Failed in This Context

While tabular Q-learning is effective for:
	•	Small and discrete environments,
	•	Stable dynamics,
	•	Dense rewards,
	•	Repeatable interactions (e.g., games or grid worlds),

it struggles when applied to:
	•	High-dimensional, real-world data,
	•	Continually drifting input distributions,
	•	Sparse feedback loops (e.g., anomaly detection),
	•	Limited exploration episodes.

In our case, the agent lacked the expressiveness to handle the subtlety and scale of the CICIDS2018 batches.

⸻

6. Conclusion and Future Directions

Our experiments demonstrate that tabular Q-learning is ill-suited for batch-wise adaptive thresholding in non-stationary, high-volume intrusion detection tasks. Despite thoughtful design of metrics and reward structures, the approach collapsed into local plateaus and failed to dynamically respond to daily distributional shifts or rare-event dynamics.

Future Work:
	•	Deep Q-Network (DQN): Enables richer state input (e.g., anchor magnitude, attack ratio, recent batch trends).
	•	Policy Gradient Methods: Support continuous actions and smooth reward adaptation.
	•	Replay Memory and Target Networks: Provide experience reuse and temporal stability.
	•	Multi-Metric States: Incorporate statistical batch summaries (e.g., KS drift, class skew) into the state definition.

Our ongoing efforts aim to integrate these strategies to realize a more intelligent, self-correcting thresholding system capable of high-precision anomaly detection in dynamic network environments.

⸻

For questions, feedback, or collaboration inquiries, please contact the development team or open an issue on the repository.
