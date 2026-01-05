import torch
import torch.nn as nn
import torch.optim as optim
from src.training.cnn_encoder import CNNEncoder
from src.utils.config import LEARNING_RATE, PPO_CLIP, ENTROPY_COEF, VALUE_LOSS_COEF

class ActorCritic(nn.Module):
    def __init__(self, image_shape=(1, 64, 64), state_dim=3, action_dim=3):
        super(ActorCritic, self).__init__()
        
        self.encoder = CNNEncoder(input_shape=image_shape)
        
        # State vector path (Balance, Shares, Portfolio Value)
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        
        # Combined features
        combined_dim = 512 + 64
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image, state):
        img_features = self.encoder(image)
        state_features = self.state_fc(state)
        combined = torch.cat([img_features, state_features], dim=1)
        
        probs = self.actor(combined)
        value = self.critic(combined)
        return probs, value

class PPOAgent:
    def __init__(self, image_shape=(1, 64, 64), state_dim=3, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.policy = ActorCritic(image_shape, state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(image_shape, state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, image, state):
        with torch.no_grad():
            image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, _ = self.policy_old(image, state)
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # Implementation of PPO update logic
        # (Simplified for the implementation plan context, full logic in train.py or here)
        pass

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
