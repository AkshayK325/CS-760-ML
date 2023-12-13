import numpy as np

# Define the parameters for the Q-learning problem
states = ["A", "B"]
actions = ["move", "stay"]
alpha = 0.5
gamma = 0.8
epsilon = 0.5
episodes = 200

# Initialize Q-table with zeros for both deterministic and epsilon-greedy approaches
Q_deterministic = np.zeros((len(states), len(actions)))
Q_epsilon_greedy = np.zeros((len(states), len(actions)))

# Define the reward function
def reward(state, action):
    if action == "move":
        return 0
    else:
        # If the action is "stay"
        return 1 if state == "A" else 1

# Define a function to choose the action based on the current Q-table
def choose_action(state_index, Q, epsilon=0):
    # Epsilon-greedy policy
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))  # Explore: random action
    else:
        # Exploit: best action from Q-table
        best_actions = np.argwhere(Q[state_index] == np.amax(Q[state_index])).flatten()
        return np.random.choice(best_actions) if len(best_actions) > 1 else best_actions[0]

# Q-learning algorithm
def q_learning(Q, episodes, alpha, gamma, epsilon=0):
    for _ in range(episodes):
        # Start in state A
        state_index = 0  # Index of state A

        # Choose an action using the policy derived from Q
        action_index = choose_action(state_index, Q, epsilon)

        # Take action and observe reward and new state
        r = reward(states[state_index], actions[action_index])
        new_state_index = 1 - state_index if actions[action_index] == "move" else state_index

        # Q-learning update
        Q[state_index, action_index] = (1 - alpha) * Q[state_index, action_index] + \
                                       alpha * (r + gamma * np.max(Q[new_state_index]))

# Run Q-learning with deterministic greedy policy
q_learning(Q_deterministic, episodes, alpha, gamma)

# Run Q-learning with epsilon-greedy policy
q_learning(Q_epsilon_greedy, episodes, alpha, gamma, epsilon)

# Print the final Q-tables
print("Q-table for deterministic greedy policy:")
print(Q_deterministic)
print("\nQ-table for epsilon-greedy policy:")
print(Q_epsilon_greedy)
