from src.gomoku.game import GomokuGame

def test_basic_game_flow():
    # 创建新游戏
    game = GomokuGame()
    
    # 初始状态检查
    assert game.current_player == 1  # 黑子先手
    assert not game.game_over
    assert game.winner is None
    
    # 第一步：黑子下在中心点
    assert game.make_move(7, 7) == True
    assert game.board[7][7] == 1  # 确认位置有黑子
    assert game.current_player == 2  # 轮到白子
    
    # 第二步：白子下在旁边
    assert game.make_move(7, 8) == True
    assert game.board[7][8] == 2  # 确认位置有白子
    assert game.current_player == 1  # 轮到黑子
    
    print("基本游戏流程测试通过！") 