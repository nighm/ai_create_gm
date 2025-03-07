from src.gomoku.game import GomokuGame
from src.gomoku.ai_model import GomokuAI
import numpy as np

def print_board_state(state_tensor):
    """打印棋盘状态的三个通道"""
    state = state_tensor.squeeze(0).numpy()
    
    print("\n当前玩家的棋子 (通道1):")
    print(state[0].astype(int))
    
    print("\n对手的棋子 (通道2):")
    print(state[1].astype(int))
    
    print("\n空位置 (通道3):")
    print(state[2].astype(int))

def test_state_representation():
    game = GomokuGame()
    ai = GomokuAI()
    
    # 模拟几步棋
    moves = [(7, 7), (7, 8), (8, 7)]
    for i, (row, col) in enumerate(moves):
        game.make_move(row, col)
    
    # 获取并显示状态
    state = ai.get_state_tensor(game.board, game.current_player)
    print("\n=== 棋局状态可视化 ===")
    print_board_state(state)
    print("\n当前玩家:", "黑子" if game.current_player == 1 else "白子")

if __name__ == "__main__":
    test_state_representation() 