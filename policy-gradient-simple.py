import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt

# Determine if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Policy Network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # We use softmax to get a probability distribution over actions
        x = F.softmax(self.layer2(x), dim=1)
        return x

class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 1e-3
        
        # Policy network and optimizer
        self.policy = Policy(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Memory for the current episode
        self.saved_log_probs = []
        self.rewards = []

    def act(self, state):
        """Chooses an action based on the policy network's output probabilities."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get action probabilities from the network
        action_probs = self.policy(state)
        
        # Create a probability distribution and sample an action
        m = Categorical(action_probs)
        action = m.sample()
        
        # Save the log probability of the chosen action (needed for the loss calculation)
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()

    def update(self):
        """Updates the policy network at the end of an episode."""
        R = 0
        policy_loss = []
        discounted_rewards = []

        # Calculate discounted rewards, iterating backwards from the end of the episode
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
            
        # Normalize the rewards for more stable training
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate the loss for each step in the episode
        for log_prob, R in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)

        # Perform the update
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() # Sum the loss for the entire episode
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear the memory for the next episode
        self.saved_log_probs = []
        self.rewards = []


# --- Main Training Loop ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PolicyGradientAgent(state_size, action_size)
    
    EPISODES = 1500
    
    scores = []
    scores_window = deque(maxlen=100)

    for e in range(1, EPISODES + 1):
        state, _ = env.reset()
        score = 0
        
        while True:
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            agent.rewards.append(reward)
            score += 1
            if done:
                break
        
        scores.append(score)
        scores_window.append(score)
        
        agent.update() # Perform the learning step at the end of the episode
        
        print(f'\rEpisode {e}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if e % 100 == 0:
            print(f'\rEpisode {e}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= 475.0:
            print(f'\nEnvironment solved in {e:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            break

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    rolling_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(rolling_avg, color='red', linewidth=3, label='100-episode average')
    plt.title("REINFORCE Agent Performance on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()