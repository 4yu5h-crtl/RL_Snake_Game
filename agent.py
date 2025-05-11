import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Start with high randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 3)  # Updated input size to 16 to match actual state size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.last_actions = deque(maxlen=10)  # Track last actions to detect patterns
        self.epsilon_decay = 0.999  # Even slower decay for more exploration
        self.min_epsilon = 0.01  # Minimum exploration rate

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Manhattan distance to food (normalized)
        max_dist = game.w + game.h
        manhattan_dist = (abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)) / max_dist

        # Calculate if food is in the same row or column as the head
        food_same_row = game.head.y == game.food.y
        food_same_col = game.head.x == game.food.x

        # Calculate if there's a clear path to food in each direction
        clear_path_left = not any(p.x < game.head.x and p.y == game.head.y for p in game.snake[1:])
        clear_path_right = not any(p.x > game.head.x and p.y == game.head.y for p in game.snake[1:])
        clear_path_up = not any(p.y < game.head.y and p.x == game.head.x for p in game.snake[1:])
        clear_path_down = not any(p.y > game.head.y and p.x == game.head.x for p in game.snake[1:])

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            # Additional state information
            manhattan_dist,  # normalized distance to food
            food_same_row,   # food in same row
            food_same_col,   # food in same column
            clear_path_left, # clear path to food on left
            clear_path_right # clear path to food on right
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Track last actions - convert numpy array to tuple for hashability
        self.last_actions.append(tuple(state))
        
        # Check for repetitive patterns
        if len(self.last_actions) >= 5:
            # If we're repeating the same state too often, increase exploration
            if len(set(self.last_actions)) < 3:
                self.epsilon = min(1.0, self.epsilon + 0.1)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            final_move = [0, 0, 0]
            final_move[random.randint(0, 2)] = 1
        else:
            # Predicted action
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            final_move = [0, 0, 0]
            final_move[torch.argmax(prediction).item()] = 1

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return final_move 