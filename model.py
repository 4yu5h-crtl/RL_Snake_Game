import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)  # Additional hidden layer
        self.linear3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)  # Add weight decay for regularization
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # If we have a batch of data
        if len(state.shape) == 2:
            # Get predicted Q values with current states
            pred = self.model(state)
            
            # Get target Q values
            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                target[idx][torch.argmax(action[idx]).item()] = Q_new
        else:
            # Single state
            pred = self.model(state)
            target = pred.clone()
            Q_new = reward
            if not done:
                Q_new = reward + self.gamma * torch.max(self.model(next_state))
            target[torch.argmax(action).item()] = Q_new
        
        # Calculate loss and update model
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step() 