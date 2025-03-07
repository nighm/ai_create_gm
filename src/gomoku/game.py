from typing import List, Tuple, Optional
import numpy as np

class GomokuGame:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for black, 2 for white
        self.last_move = None
        self.game_over = False
        self.winner = None

    def is_valid_move(self, row: int, col: int) -> bool:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        return self.board[row][col] == 0

    def make_move(self, row: int, col: int) -> bool:
        if not self.is_valid_move(row, col) or self.game_over:
            return False
        
        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif np.count_nonzero(self.board) == self.board_size * self.board_size:
            self.game_over = True
            self.winner = 0  # Draw
        
        self.current_player = 3 - self.current_player  # Switch player (1->2 or 2->1)
        return True

    def check_win(self, row: int, col: int) -> bool:
        player = self.board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check forward direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check backward direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        return False

    def get_state(self) -> dict:
        return {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "last_move": self.last_move
        }

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.game_over = False
        self.winner = None 