import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Same DQN Agent and QNetwork Class as Before ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Main Training Loop ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 300 # Reduced episodes for a quicker demonstration
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQUENCY = 10
    
    # List to store scores for plotting
    scores = []

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
        
        scores.append(time + 1)
        print(f"Episode: {e+1}/{EPISODES}, Score: {time+1}, Epsilon: {agent.epsilon:.2f}")

        agent.replay(BATCH_SIZE)
        
        if e % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_model()

    print("\nTraining finished.\n")

    # === VISUALIZATION 1: Plotting the Learning Curve ===
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("DQN Agent Performance on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Score (Timesteps Survived)")
    plt.grid(True)
    plt.show()

    # === VISUALIZATION 2: Watching the Trained Agent ===
    print("Displaying trained agent for 3 episodes...")
    # Use a new environment with "human" render mode
    env = gym.make("CartPole-v1", render_mode="human")
    for e in range(3):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # Tell the agent to act greedily (no exploration)
            action = agent.act(state, use_epsilon=False)
            state, _, done, _, _ = env.step(action)
            state = np.reshape(state, [1, state_size])
            if done:
                print(f"Episode {e+1} finished after {time+1} timesteps")
                break
    env.close()

    # === VISUALIZATION 3: The Policy Landscape (Q-Value Heatmap) ===
    print("\nGenerating Policy Landscape visualization...")
    # We create a grid for two state variables: Pole Angle and Pole Velocity
    # We hold the other two (Cart Position, Cart Velocity) constant at 0
    pole_angle_range = np.linspace(-0.2095, 0.2095, 100) # Range of pole angles
    pole_velocity_range = np.linspace(-2.0, 2.0, 100) # Range of pole velocities
    
    policy_map = np.zeros((len(pole_angle_range), len(pole_velocity_range)))

    for i, angle in enumerate(pole_angle_range):
        for j, velocity in enumerate(pole_velocity_range):
            # State: [cart_pos, cart_vel, pole_angle, pole_vel]
            state = np.array([[0.0, 0.0, angle, velocity]])
            # We store the preferred action (0 for left, 1 for right)
            policy_map[i, j] = agent.act(state, use_epsilon=False)

    plt.figure(figsize=(10, 8))
    plt.imshow(policy_map.T, extent=[-0.2095, 0.2095, -2.0, 2.0], aspect='auto', origin='lower')
    plt.colorbar(ticks=[0, 1], label="Action (0=Left, 1=Right)")
    plt.title("Policy Landscape for CartPole Agent")
    plt.xlabel("Pole Angle (radians)")
    plt.ylabel("Pole Angular Velocity (rad/s)")
    plt.show()