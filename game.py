import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.last_positions = []  # Track last positions to detect loops
        self.max_positions = 10   # Number of positions to track
        self.steps_without_food = 0  # Track steps without eating food
        self.last_food_distance = 0  # Track distance to food

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.last_positions = []  # Reset position history
        self.steps_without_food = 0
        self.last_food_distance = self._get_food_distance()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Process events to prevent GIL issues
        pygame.event.pump()
        
        # Store current position before moving
        current_position = self.head
        
        self._move(action)
        self.snake.insert(0, self.head)

        # Base reward is small negative to encourage efficiency
        reward = -0.1
        game_over = False

        # Check for collision
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -20  # Penalty for dying
            return reward, game_over, self.score

        # Check if snake ate food
        if self.head == self.food:
            self.score += 1
            reward = 20  # Reward for eating food
            self._place_food()
            self.steps_without_food = 0  # Reset steps counter
            self.last_food_distance = self._get_food_distance()
        else:
            self.snake.pop()
            self.steps_without_food += 1
            
            # Calculate current distance to food
            current_food_distance = self._get_food_distance()
            
            # Reward for moving closer to food
            if current_food_distance < self.last_food_distance:
                reward += 0.5
            # Penalty for moving away from food
            elif current_food_distance > self.last_food_distance:
                reward -= 0.5
                
            # Update last food distance
            self.last_food_distance = current_food_distance
            
            # Penalty for taking too many steps without food
            if self.steps_without_food > 50:
                reward -= 0.2

        # Check for repetitive patterns
        self.last_positions.append(self.head)
        if len(self.last_positions) > self.max_positions:
            self.last_positions.pop(0)
            
        # Penalize if snake is moving in circles
        if self._is_moving_in_circles():
            reward -= 2.0

        self._update_ui()
        
        # Add a small delay to prevent GIL issues
        pygame.time.delay(10)
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw path to food (visual aid for learning)
        if len(self.snake) > 0:
            # Draw a line from head to food
            pygame.draw.line(self.display, (100, 100, 100), 
                           (self.head.x + BLOCK_SIZE/2, self.head.y + BLOCK_SIZE/2),
                           (self.food.x + BLOCK_SIZE/2, self.food.y + BLOCK_SIZE/2), 1)
        
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _is_moving_towards_food(self, old_pos, new_pos):
        """Check if the snake is moving towards the food"""
        old_dist = abs(old_pos.x - self.food.x) + abs(old_pos.y - self.food.y)
        new_dist = abs(new_pos.x - self.food.x) + abs(new_pos.y - self.food.y)
        return new_dist < old_dist

    def _is_moving_in_circles(self):
        """Check if the snake is moving in circles"""
        if len(self.last_positions) < 4:
            return False
            
        # Check if the last few positions form a small area
        x_coords = [p.x for p in self.last_positions]
        y_coords = [p.y for p in self.last_positions]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # If the snake is staying in a small area, it might be circling
        return x_range < BLOCK_SIZE * 3 and y_range < BLOCK_SIZE * 3
        
    def _get_food_distance(self):
        """Calculate Manhattan distance to food"""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y) 