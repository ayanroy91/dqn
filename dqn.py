
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, rows, cols, magic_squares):
        self.grid = np.zeros((rows, cols))
        self.rows = rows
        self.cols = cols
        self.total_states = rows * cols
        self.state_space = list(range(self.total_states))
        self.state_space.remove(80)
        self.state_space_plus = list(range(self.total_states))
        self.action_map = {'U': -self.rows, 'D': self.rows, 'L': -1, 'R': 1}
        self.actions = ['U', 'D', 'L', 'R']
        self.add_magic_squares(magic_squares)
        self.agent_position = 0

    def is_terminal_state(self, state):
        return state not in self.state_space

    def add_magic_squares(self, magic_squares):
        self.magic_squares = magic_squares
        label = 2
        for start, end in magic_squares.items():
            start_x, start_y = divmod(start, self.cols)
            end_x, end_y = divmod(end, self.cols)
            self.grid[start_x, start_y] = label
            self.grid[end_x, end_y] = label + 1
            label += 2

    def get_agent_position(self):
        return divmod(self.agent_position, self.cols)

    def set_state(self, state):
        current_x, current_y = self.get_agent_position()
        self.grid[current_x, current_y] = 0
        self.agent_position = state
        new_x, new_y = self.get_agent_position()
        self.grid[new_x, new_y] = 1

    def is_off_grid(self, new_state, old_state):
        if new_state not in self.state_space_plus:
            return True
        elif old_state % self.cols == 0 and new_state % self.cols == self.cols - 1:
            return True
        elif old_state % self.cols == self.cols - 1 and new_state % self.cols == 0:
            return True
        return False

    def step(self, action):
        agent_x, agent_y = self.get_agent_position()
        new_state = self.agent_position + self.action_map[action]
        if new_state in self.magic_squares:
            new_state = self.magic_squares[new_state]
        
        reward = -1 if not self.is_terminal_state(new_state) else 0
        if not self.is_off_grid(new_state, self.agent_position):
            self.set_state(new_state)
            return new_state, reward, self.is_terminal_state(new_state), None
        else:
            return self.agent_position, reward, self.is_terminal_state(self.agent_position), None

    def reset(self):
        self.agent_position = 0
        self.grid.fill(0)
        self.add_magic_squares(self.magic_squares)
        return self.agent_position

    def render(self):
        print('------------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
        print('------------------------------------------')

    def sample_action(self):
        return np.random.choice(self.actions)

def get_best_action(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    best_action = np.argmax(values)
    return actions[best_action]

if __name__ == '__main__':
    magic_squares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magic_squares)
    
    ALPHA = 0.1
    GAMMA = 1.0
    EPSILON = 1.0
    
    Q = {}
    for state in env.state_space_plus:
        for action in env.actions:
            Q[state, action] = 0
    
    num_episodes = 50000
    rewards = np.zeros(num_episodes)
    env.render()
    
    for episode in range(num_episodes):
        if episode % 5000 == 0:
            print(f'Starting episode {episode}')
        done = False
        total_reward = 0
        state = env.reset()
        
        while not done:
            if np.random.random() < EPSILON:
                action = env.sample_action()
            else:
                action = get_best_action(Q, state, env.actions)
                
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            best_next_action = get_best_action(Q, new_state, env.actions)
            Q[state, action] += ALPHA * (reward + GAMMA * Q[new_state, best_next_action] - Q[state, action])
            state = new_state
        
        EPSILON = max(EPSILON - 2 / num_episodes, 0)
        rewards[episode] = total_reward

    plt.plot(rewards)
    plt.show()
