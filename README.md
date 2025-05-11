# 🐍 Snake Game with Reinforcement Learning 🤖

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)

## 📋 Overview

This project implements a classic Snake game where an AI agent learns to play using Deep Q-Learning. The agent gradually improves its strategy through reinforcement learning, learning to navigate the game environment, collect food, and avoid collisions.

## 🚀 Features

- 🎮 Classic Snake gameplay with Pygame visualization
- 🧠 Deep Q-Learning implementation with PyTorch
- 📊 Real-time training visualization with Matplotlib
- 🔄 Adaptive exploration strategy with epsilon decay
- 💾 Model checkpointing and saving
- 🧩 Advanced state representation for better learning
- 🏆 Reward system designed to encourage efficient gameplay

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL_Snake_Game.git
cd RL_Snake_Game
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

Run the training script to start the AI learning process:

```bash
python main.py
```

The training progress will be displayed in real-time with a graph showing the score and mean score over time.

## 🧮 How It Works

### 🧠 Neural Network Architecture
- Input layer: 16 features representing the game state
- Hidden layers: 256 and 128 neurons with ReLU activation
- Output layer: 3 neurons (straight, right, left movements)
- Dropout for regularization

### 📊 State Representation
The agent observes:
- Danger in three directions (straight, right, left)
- Current movement direction
- Food location relative to the snake
- Distance to food
- Clear paths to food

### 🏆 Reward System
- +20 for eating food
- -20 for collision
- +0.5 for moving closer to food
- -0.5 for moving away from food
- -0.1 base reward to encourage efficiency
- -2.0 for moving in circles

## 📈 Training Process

The agent uses an epsilon-greedy strategy for exploration:
- Starts with high exploration (epsilon = 1.0)
- Gradually reduces exploration as it learns
- Resets exploration if performance is poor
- Uses experience replay for stable learning

## 📂 Project Structure

- `main.py` - Entry point and training loop
- `game.py` - Snake game implementation
- `agent.py` - AI agent with Q-learning logic
- `model.py` - Neural network model
- `helper.py` - Utility functions for plotting

## 🔮 Future Improvements

- [ ] Implement prioritized experience replay
- [ ] Add convolutional layers for visual input
- [ ] Create a human-playable mode
- [ ] Add difficulty levels
- [ ] Implement multi-agent training