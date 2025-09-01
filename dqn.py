import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CHANGED: Slightly larger network for more capacity ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000) # Increased buffer size
        
        # --- TUNED HYPERPARAMETERS ---
        self.gamma = 0.99          # More farsighted
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Slower decay
        self.learning_rate = 5e-4    # Slightly lower learning rate
        self.tau = 1e-3            # For soft update of target parameters
        
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Initialize target model with same weights
        self.target_model.load_state_dict(self.model.state_dict())

    # --- NEW: Soft update method ---
    def soft_update_target_model(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): # No change here, but epsilon decay is slower
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in minibatch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in minibatch]).astype(np.uint8)).float().to(device)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
        predicted_q_values = self.model(states).gather(1, actions)
        loss = self.loss_fn(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # --- CHANGED: Now we call the soft update every time we learn ---
        self.soft_update_target_model()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Main Training Loop ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 600 # Increased episodes to see full learning curve
    BATCH_SIZE = 64 # Larger batch size for more stable gradients
    
    scores = []
    scores_window = deque(maxlen=100) # For tracking average score over last 100 episodes

    for e in range(1, EPISODES + 1):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(BATCH_SIZE) # Learn at every step
            
            state = next_state
            score += 1
            if done:
                break
        
        scores.append(score)
        scores_window.append(score)
        
        print(f'\rEpisode {e}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if e % 100 == 0:
            print(f'\rEpisode {e}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= 475.0: # A common success criterion
            print(f'\nEnvironment solved in {e:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            break

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    # Also plot a rolling average to see the trend more clearly
    rolling_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(rolling_avg, color='red', linewidth=3, label='100-episode average')
    plt.title("DQN Agent Performance on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()