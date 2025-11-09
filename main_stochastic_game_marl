import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Random action initialization
def random_action_init(payoff_matrix):
    possible_actions_a1 = payoff_matrix['rows'].unique()
    possible_actions_a2 = payoff_matrix['cols'].unique()
    a1 = np.random.choice(possible_actions_a1)
    a2 = np.random.choice(possible_actions_a2)
    return a1, a2

# Log-linear learning rule
def log_linear_update(q_values, epsilon):
    q_values = np.array(q_values)
    max_q = np.max(q_values)
    exp_q = np.exp((q_values - max_q) / epsilon)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_values), p=probs), probs

# Retrieve payoffs
def get_payoff(action1, action2, payoff_matrix):
    row = payoff_matrix[(payoff_matrix['rows'] == action1) & (payoff_matrix['cols'] == action2)]
    if not row.empty:
        return row['payoffs'].values[0]
    return None

# Simulate learning and estimate Nash equilibrium
def algo_with_nash_estimation(payoff_matrix, T, runs, epsilon=0.1, alpha=0.1):
    action_pairs = payoff_matrix[['rows', 'cols']].values
    Q = np.zeros((len(action_pairs)))  # Q-values for all action pairs
    V = np.zeros((len(action_pairs)))  # Value function for all states
    policy_counts = {tuple(action): np.zeros(T) for action in action_pairs}  # Track action counts over time

    for run in range(runs):
        Q = np.zeros((len(action_pairs)))  # Reset Q-values for each run
        V = np.zeros((len(action_pairs)))  # Reset value function for each run
        current_action = random_action_init(payoff_matrix)

        for t in range(1, T + 1):
            # Get the payoff for the current action
            reward = get_payoff(current_action[0], current_action[1], payoff_matrix)

            if reward:
                # Update the value function
                action_index = np.where((action_pairs == current_action).all(axis=1))[0][0]
                V[action_index] = (t / (t + 1)) * V[action_index] + (1 / (t + 1)) * Q[action_index]

                # Update Q-values using the updated value function
                Q[action_index] = reward[0] + V[action_index]

            # Update policy probabilities using log-linear learning
            probs = log_linear_update(Q, epsilon)[1]

            # Count action occurrences
            current_action_tuple = tuple(current_action)
            if current_action_tuple in policy_counts:
                policy_counts[current_action_tuple][t - 1] += 1

            # Select the next action
            next_action_index = np.random.choice(len(action_pairs), p=probs)
            current_action = action_pairs[next_action_index]

    # Normalize policy counts to estimate probabilities
    for action in policy_counts:
        policy_counts[action] /= runs

    return policy_counts

# Parameters
payoff_matrix = pd.DataFrame({
    'rows': ['a1 = 0', 'a1 = 0', 'a1 = 1', 'a1 = 1'],
    'cols': ['a2 = 0', 'a2 = 1', 'a2 = 0', 'a2 = 1'],
    'payoffs': [(0, 0), (0, 2), (2, 0), (1, 1)]
}) 
T = 10000
runs = 100
epsilon = 0.001
alpha = 0.01

# Run the algorithm
policy_counts = algo_with_nash_estimation(payoff_matrix, T, runs, epsilon, alpha)

# Plot the results
plt.figure(figsize=(10, 6))

# Extract indices for (Right, Up) = (0, 0) and (Up, Left) = (1, 1)
action_labels = list(zip(payoff_matrix['rows'], payoff_matrix['cols']))
action_00_index = ('a1 = 0', 'a2 = 0')  # Action pair (a1 = 0, a2 = 0)
action_11_index = ('a1 = 1', 'a2 = 1')  # Action pair (a1 = 1, a2 = 1)

# Plot action probabilities over time
timesteps = np.arange(1, T + 1)
plt.plot(timesteps, policy_counts[action_00_index], label=r'$(\text{a1 = 0}, \text{a2 = 0})$', color='red')
plt.plot(timesteps, policy_counts[action_11_index], label=r'$(\text{a1 = 1}, \text{a2 = 1})$', color='green')

# Add shaded regions for uncertainty
plt.fill_between(timesteps,
                 policy_counts[action_00_index] - 0.1 * policy_counts[action_00_index],
                 policy_counts[action_00_index] + 0.1 * policy_counts[action_00_index],
                 color='red', alpha=0.2)
plt.fill_between(timesteps,
                 policy_counts[action_11_index] - 0.1 * policy_counts[action_11_index],
                 policy_counts[action_11_index] + 0.1 * policy_counts[action_11_index],
                 color='green', alpha=0.2)

# Final plot adjustments
plt.title("Evolution of Policy Probabilities (Empirical Frequencies)")
plt.xlabel("Timesteps")
plt.ylabel(r"$\pi^t(a|s)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
