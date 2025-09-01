import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os

"""
DQN Agent for CartPole (PyTorch Version)
----------------------------------------
This agent uses a PyTorch neural network to approximate the Q-function.
"""

# Determine if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Neural Network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Linear output for Q-values

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Replay Buffer
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Neural Networks
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_model()
        
        # Optimizer and Loss Function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action using the Epsilon-Greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Put model in evaluation mode and disable gradients for inference
        self.model.eval()
        with torch.no_grad():
            # Convert state to a PyTorch tensor and send to the correct device
            state_tensor = torch.from_numpy(state).float().to(device)
            act_values = self.model(state_tensor)
        self.model.train() # Put model back in training mode
        
        # Choose the action with the highest Q-value
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        """Trains the main model using a random sample of experiences."""
        if len(self.memory) < batch_size:
            return
        
        # 1. Sample a random mini-batch of experiences
        minibatch = random.sample(self.memory, batch_size)
        
        # 2. Unzip experiences and convert to PyTorch tensors
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in minibatch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in minibatch]).astype(np.uint8)).float().to(device)
        
        # 3. Calculate the target Q-values (using the target network for stability)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            # Bellman equation: target = reward + gamma * max_q' * (1 - done)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 4. Get the Q-values predicted by the main model for the actions that were taken
        predicted_q_values = self.model(states).gather(1, actions)

        # 5. Compute the loss between predicted and target Q-values
        loss = self.loss_fn(predicted_q_values, target_q_values)
        
        # 6. Perform the optimization step
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()            # Backpropagate the loss
        self.optimizer.step()      # Update the weights
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Main Training Loop ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 500
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQUENCY = 10

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        print(f"Episode: {e+1}/{EPISODES}, Score: {time+1}, Epsilon: {agent.epsilon:.2f}")

        agent.replay(BATCH_SIZE)
        
        if e % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_model()

    print("\nTraining finished.")