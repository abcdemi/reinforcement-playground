import numpy as np
import random
import time

# 1. Define the Environment and its Rules
# =======================================

# Define the states (grid cells)
# 0 1 2
# 3 4 5  (State 4 is the Wall)
# 6 7 8
states = list(range(9))
terminal_states = [5, 8]  # Trap and Cheese

# Define the actions (0: up, 1: down, 2: left, 3: right)
actions = [0, 1, 2, 3]

# Define the rewards for each state
# We use a dictionary where the key is the state and value is the reward
rewards = {
    0: -1, 1: -1, 2: -1,
    3: -1, 4: -1, 5: -10, # State 5 is the Trap
    6: -1, 7: -1, 8: 10   # State 8 is the Cheese
}

# Define the transitions (the rules of movement)
# Dictionary: key = (current_state, action), value = next_state
# This defines the deterministic movement in our grid.
transitions = {
    (0, 1): 3, (0, 3): 1,  # From state 0, can go down to 3 or right to 1
    (1, 1): 4, (1, 2): 0, (1, 3): 2,
    (2, 1): 5, (2, 2): 1,
    (3, 0): 0, (3, 1): 6, (3, 3): 4,
    # State 4 is a wall, no movement from it in this simple model
    # State 5 is a trap (terminal)
    (6, 0): 3, (6, 3): 7,
    (7, 0): 4, (7, 2): 6, (7, 3): 8,
    # State 8 is cheese (terminal)
}

# 2. Initialize the Q-Table
# ===========================
# Create a table of States x Actions, initialized to zeros
q_table = np.zeros((len(states), len(actions)))

# 3. Set the Hyperparameters for the Q-Learning Algorithm
# ========================================================
learning_rate = 0.1      # Alpha: How much we update Q-values based on new info
discount_factor = 0.9    # Gamma: Importance of future rewards
epsilon = 1.0            # Initial exploration rate
epsilon_decay_rate = 0.001 # Rate at which epsilon decreases
min_epsilon = 0.01       # Minimum exploration rate

num_episodes = 1000      # How many times the agent will play the game

# 4. The Q-Learning Algorithm Loop
# ================================
for episode in range(num_episodes):
    # Start each episode from the beginning
    current_state = 0
    done = False

    while not done:
        # Action selection: Epsilon-Greedy strategy
        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = random.choice(actions)
        else:
            # Exploit: choose the best action from Q-table
            action = np.argmax(q_table[current_state, :])

        # Get the next state based on the chosen action
        # If the move is invalid (e.g., into a wall or off-grid), stay in the same state
        next_state = transitions.get((current_state, action), current_state)

        # Get the reward for the new state
        reward = rewards.get(next_state, 0)
        
        # Check if the episode is finished
        if next_state in terminal_states:
            done = True

        # Q-value update rule (The Bellman Equation)
        old_q_value = q_table[current_state, action]
        max_future_q = np.max(q_table[next_state, :]) # Best Q-value for the next state

        # The core Q-learning formula
        new_q_value = old_q_value + learning_rate * (reward + discount_factor * max_future_q - old_q_value)
        q_table[current_state, action] = new_q_value

        # Move to the next state
        current_state = next_state

    # Decay epsilon after each episode (less exploration, more exploitation)
    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

# 5. Display the Results
# ======================
print("Training finished.\n")
print("Final Q-Table:")
print(q_table)

# Let's see the learned policy by finding the best path
print("\nOptimal Path from Start (S):")
current_state = 0
path = [current_state]
while current_state not in terminal_states:
    # Choose the best action from the learned Q-table
    action = np.argmax(q_table[current_state, :])
    # Get the next state
    next_state = transitions.get((current_state, action), current_state)
    path.append(next_state)
    current_state = next_state
    # Safety break to prevent infinite loops if something goes wrong
    if len(path) > 10:
        print("Path seems too long, breaking.")
        break
        
print(path)