import matplotlib.pyplot as plt
import time

plt.ion()

def plot(scores, mean_scores):
    try:
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.draw()
        plt.pause(0.1)
        time.sleep(0.01)
    except Exception as e:
        print(f"Plotting error: {e}") 