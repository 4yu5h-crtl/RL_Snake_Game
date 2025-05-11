from agent import Agent
from game import SnakeGameAI
from helper import plot
import time
import os
import torch


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Set a higher initial exploration rate
    agent.epsilon = 1.0
    
    # Set a slower decay rate for better exploration
    agent.epsilon_decay = 0.995
    
    # Create model directory if it doesn't exist
    if not os.path.exists('./model'):
        os.makedirs('./model')
    
    # Try to load a previous model if it exists
    try:
        agent.model.load_state_dict(torch.load('./model/model.pth'))
        print("Loaded previous model")
    except:
        print("Starting with a new model")
    
    # Training loop
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', round(agent.epsilon, 3))

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            # Reset exploration rate if score is too low
            if score < 2 and agent.n_games > 10:
                agent.epsilon = min(1.0, agent.epsilon + 0.1)
                
            # Save model periodically
            if agent.n_games % 50 == 0:
                agent.model.save(f'model_checkpoint_{agent.n_games}.pth')
                
            # Add a small delay to prevent GIL issues
            time.sleep(0.01)

if __name__ == '__main__':
    train() 