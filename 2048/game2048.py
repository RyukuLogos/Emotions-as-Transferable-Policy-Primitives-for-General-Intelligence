import numpy as np
import random

class Game2048:
    """2048 game environment"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game"""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self._add_random_tile()
        self._add_random_tile()
        return self._get_state()
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4
    
    def _compress(self, row):
        """Remove zeros from row"""
        new_row = [i for i in row if i != 0]
        new_row += [0] * (4 - len(new_row))
        return new_row
    
    def _merge(self, row):
        """Merge adjacent equal tiles"""
        for i in range(3):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                self.score += row[i]
                self.max_tile = max(self.max_tile, row[i])
                row[i + 1] = 0
        return row
    
    def _move_left(self):
        """Move all tiles left"""
        old_board = self.board.copy()
        for i in range(4):
            row = self.board[i, :]
            row = self._compress(row)
            row = self._merge(row)
            row = self._compress(row)
            self.board[i, :] = row
        return not np.array_equal(old_board, self.board)
    
    def step(self, action):
        """
        Execute action: 0=up, 1=right, 2=down, 3=left
        Returns: state, reward, done
        """
        old_score = self.score
        old_max = self.max_tile
        
        # Rotate board to convert all moves to left
        if action == 0:  # up
            self.board = np.rot90(self.board, k=1)
            moved = self._move_left()
            self.board = np.rot90(self.board, k=-1)
        elif action == 1:  # right
            self.board = np.rot90(self.board, k=2)
            moved = self._move_left()
            self.board = np.rot90(self.board, k=-2)
        elif action == 2:  # down
            self.board = np.rot90(self.board, k=-1)
            moved = self._move_left()
            self.board = np.rot90(self.board, k=1)
        elif action == 3:  # left
            moved = self._move_left()
        
        # Add new tile if board changed
        if moved:
            self._add_random_tile()
        
        # Calculate reward
        reward = self._calculate_reward(old_max)
        
        # Check game over
        done = self._is_game_over()
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self, old_max):
        """Calculate reward based on tile achievement"""
        reward = 0.0
        
        # Score increase
        reward += (self.score - old_max) / 10000.0
        
        # Milestone rewards for reaching new tiles
        if self.max_tile > old_max:
            tile_rewards = {
                128: 0.1,
                256: 0.2,
                512: 0.3,
                1024: 0.5,
                2048: 0.7,
                4096: 0.85,
                8192: 0.95,
                16384: 1.0
            }
            reward += tile_rewards.get(self.max_tile, 0.0)
        
        return reward
    
    def _is_game_over(self):
        """Check if no more moves possible"""
        # Check for empty cells
        if np.any(self.board == 0):
            return False
        
        # Check for possible merges
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return False
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def _get_state(self):
        """Get current state as normalized array"""
        # Log2 normalization (0 for empty, 1 for 2, 2 for 4, etc.)
        state = np.where(self.board == 0, 0, np.log2(self.board))
        return state.astype(np.float32) / 16.0  # Normalize to [0, 1]
    
    def get_legal_actions(self):
        """Return list of legal actions"""
        legal = []
        for action in range(4):
            # Temporarily try the action
            old_board = self.board.copy()
            old_score = self.score
            old_max = self.max_tile
            
            self.step(action)
            
            # Check if board changed
            if not np.array_equal(old_board, self.board):
                legal.append(action)
            
            # Restore state
            self.board = old_board
            self.score = old_score
            self.max_tile = old_max
        
        return legal if legal else [0]  # Always return at least one action
