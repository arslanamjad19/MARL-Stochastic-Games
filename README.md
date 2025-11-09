# MARL-Stochastic-Games
Python implementation of the unified framework for equilibrium selection in stochastic games

# Equilibrium Selection for Multi-Agent Reinforcement Learning

A Python implementation of the unified framework for equilibrium selection in stochastic games, based on the research paper "Equilibrium Selection for Multi-agent Reinforcement Learning: A Unified Framework" by Zhang, Shamma, and Li (2024).

## Overview

This repository provides an independent implementation and reproduction attempt of the equilibrium selection framework presented in the original paper. The framework extends classical equilibrium selection results from normal-form games to the multi-agent reinforcement learning (MARL) setting, specifically focusing on stochastic games.

### Key Concepts

While most MARL research focuses on converging to *any* Nash equilibrium, this work addresses a critical limitation: **Nash equilibria can be non-unique with drastically different performance**. This implementation demonstrates algorithms that converge to high-quality equilibria rather than arbitrary ones.

The framework uses an actor-critic structure where:
- **Critic**: Estimates Q-functions of the stochastic game
- **Actor**: Applies classical learning rules from normal-form games with respect to the estimated Q-functions

## Features

- Implementation of log-linear learning for equilibrium selection
- Support for two-player, two-action stochastic games
- Visualization of policy evolution over time
- Empirical validation of convergence to specific equilibria

## Theoretical Background

### Equilibrium Selection Problem

In multi-agent systems, multiple Nash equilibria may exist with varying rewards. For example:
- **Potential Games**: Log-linear learning converges to potential-maximizing equilibria
- **General-Sum Games**: Specially designed learning rules can achieve Pareto-optimal equilibria

### Stochastically Stable Equilibria (SSE)

The framework leverages the concept of stochastically stable equilibria - equilibria that are robust to small perturbations (mistakes) in the learning process. As the "mistake rate" (ϵ) approaches zero, the algorithm converges to only the most stable, high-performance equilibria.

## Implementation Details

### Current Implementation

The code implements Algorithm 1 from the paper with the following components:

1. **Log-Linear Learning Rule**: Agents select actions probabilistically based on Q-values:
   ```
   P(action) ∝ exp(Q(action) / ε)
   ```

2. **Q-Value Updates**: Bellman-style iteration for critic updates:
   ```
   V^(t+1)(s) = (t/(t+1)) * V^(t)(s) + (1/(t+1)) * Q^(t)(s, a^(t)(s))
   Q^(t+1)(s,a) = r(s,a) + ∑P(s'|s,a)V^(t+1)(s')
   ```

3. **Policy Tracking**: Empirical frequency estimation of policy convergence

### Example: Stage-Hunt Game

The implementation includes a two-stage stag-hunt game example:
- **Stag (a=0)**: High reward (3.75 per player) but requires cooperation for 2 stages
- **Hare (a=1)**: Low reward (1-2) but easier to obtain

The algorithm demonstrates convergence to the risk-dominant or Pareto-optimal equilibrium depending on the learning rule used.

## Usage

```python
import numpy as np
import pandas as pd
from equilibrium_selection import algo_with_nash_estimation

# Define payoff matrix
payoff_matrix = pd.DataFrame({
    'rows': ['a1 = 0', 'a1 = 0', 'a1 = 1', 'a1 = 1'],
    'cols': ['a2 = 0', 'a2 = 1', 'a2 = 0', 'a2 = 1'],
    'payoffs': [(0, 0), (0, 2), (2, 0), (1, 1)]
})

# Run algorithm
T = 10000  # Number of timesteps
runs = 100  # Number of independent runs
epsilon = 0.001  # Exploration parameter
alpha = 0.01  # Learning rate

policy_counts = algo_with_nash_estimation(payoff_matrix, T, runs, epsilon, alpha)
```

## Results

The implementation reproduces key findings from the paper:

- **Convergence to High-Quality Equilibria**: Empirical frequencies show convergence to equilibria that maximize potential (for potential games) or social welfare (for general-sum games)
- **Stochastic Stability**: As ε → 0, only stochastically stable equilibria receive positive probability
- **Policy Evolution**: Visualization shows clear convergence patterns over thousands of iterations

## Requirements

```
numpy
matplotlib
pandas
```

## Installation

```bash
git clone https://github.com/yourusername/equilibrium-selection-marl
cd equilibrium-selection-marl
pip install -r requirements.txt
```

## Limitations and Future Work

This implementation currently:
- Focuses on two-player, two-action games
- Implements only log-linear learning (Example 1 from the paper)
- Provides asymptotic guarantees without finite-time convergence rates
- Uses model-based updates (requires transition probabilities)

**Future extensions** could include:
- Sample-based learning (Algorithm 2 from the paper)
- Additional learning rules (Pradelski-Young, Marden et al.)
- Multi-player games with larger action spaces
- Finite-time convergence analysis
- Application to practical MARL problems

## Citation

This implementation is based on the following paper:

```bibtex
@article{zhang2024equilibrium,
  title={Equilibrium Selection for Multi-agent Reinforcement Learning: A Unified Framework},
  author={Zhang, Runyu and Shamma, Jeff and Li, Na},
  journal={arXiv preprint arXiv:2406.08844},
  year={2024}
}
```

## Disclaimer

This is an **independent implementation** for a semester research project. It is not affiliated with the original authors and may differ from their implementation in various details. For the authoritative version and complete theoretical treatment, please refer to the original paper.

## Acknowledgments

- Original paper authors: Runyu (Cathy) Zhang, Jeff Shamma, and Na Li
- Classical game theory literature on equilibrium selection

## Contact

For questions or issues, please open an issue on this repository.

---

**Note**: This is a reproduction attempt and educational project. Results may vary from the original paper due to implementation details, hyperparameters, and computational resources.
