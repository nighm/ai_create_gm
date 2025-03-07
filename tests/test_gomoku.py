import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import json
import asyncio
from src.gomoku import app
from src.gomoku.game import GomokuGame
from src.gomoku.ai_model import GomokuAI

def test_game_logic():
    """
    测试基本的游戏逻辑
    """
    # 创建新游戏
    game = GomokuGame()
    
    # 测试初始状态
    assert game.current_player == 1  # 黑子先手
    assert not game.game_over
    assert game.winner is None
    
    # 测试有效移动
    assert game.make_move(7, 7) == True  # 在中心位置下一个黑子
    assert game.board[7][7] == 1  # 确认这个位置有黑子
    assert game.current_player == 2  # 现在该白子下了
    
    # 测试无效移动（已经有棋子的位置）
    assert game.make_move(7, 7) == False  # 不能在同一个位置重复下子
    
    # 测试获取有效移动
    valid_moves = game.get_valid_moves()
    assert len(valid_moves) == game.board_size * game.board_size - 1  # 总位置减去已下的一个子
    assert (7, 7) not in valid_moves  # 已经下过的位置不能再下

def test_ai_model():
    """
    测试AI模型的基本功能
    """
    # 创建AI和游戏实例
    ai = GomokuAI()
    game = GomokuGame()
    
    # 测试AI的移动
    row, col = ai.get_move(game, training=False)
    # 确保返回的位置在棋盘范围内
    assert 0 <= row < game.board_size
    assert 0 <= col < game.board_size
    # 确保是一个合法的移动
    assert game.is_valid_move(row, col)
    
    # 测试状态张量生成
    state = ai.get_state_tensor(game.board, game.current_player)
    # 确保状态张量的形状正确：[批次, 通道, 高度, 宽度]
    assert state.shape == (1, 3, game.board_size, game.board_size)

def test_win_condition():
    """
    测试游戏的胜利条件
    """
    # 创建新游戏
    game = GomokuGame()
    
    # 创建一个横向五子连线（黑子）
    for i in range(5):
        assert game.make_move(7, i) == True  # 黑子
        if i < 4:  # 白子在下面一行跟着下
            assert game.make_move(8, i) == True
    
    # 验证游戏结束，黑子胜
    assert game.game_over
    assert game.winner == 1

def test_ai_vs_ai_game():
    """
    测试AI对战功能
    
    这个测试就像是给AI安排一场练习赛：
    1. 准备比赛场地（创建游戏）
    2. 请AI开始下棋
    3. 检查每一步是否合规
    4. 确认比赛正常结束
    """
    # === 第一步：准备比赛 ===
    # 创建一个新的棋盘
    game = GomokuGame()
    # 准备一个AI选手
    ai = GomokuAI()
    
    # === 第二步：开始比赛 ===
    # 记录已经下了多少步棋
    moves_made = 0
    
    # AI开始下棋（最多下10步，这样测试不会太久）
    while not game.game_over and moves_made < 10:
        # AI思考下一步
        row, col = ai.get_move(game, training=True)
        
        # 确保这一步是合法的，并且可以成功落子
        assert game.make_move(row, col)
        moves_made += 1
    
    # === 第三步：检查比赛记录 ===
    # 获取最终的比赛状态
    state = game.get_state()
    
    # 检查比赛记录是否完整：
    assert isinstance(state, dict)  # 确保是正确的记录格式
    assert "board" in state        # 包含棋盘信息
    assert "current_player" in state  # 记录该谁下棋
    assert "game_over" in state    # 记录比赛是否结束

@pytest.mark.asyncio
async def test_websocket_ai_vs_ai():
    """
    测试网络对战功能
    
    这个测试就像是给AI安排一场网络直播比赛：
    1. 建立直播连接
    2. 发送开始比赛的信号
    3. 等待AI开始下棋
    4. 确保观众能看到每一步棋
    """
    client = TestClient(app)
    
    # === 第一步：建立直播连接 ===
    with client.websocket_connect("/ws/test_client") as websocket:
        try:
            # === 第二步：发送开始比赛的信号 ===
            data = {
                "type": "start_game",
                "mode": "ai_vs_ai"
            }
            websocket.send_json(data)
            
            # === 第三步：等待AI开始下棋 ===
            # 我们期望收到游戏状态
            response = websocket.receive_json()
            assert response["type"] == "game_state"
            assert "state" in response
            state = response["state"]
            assert isinstance(state, dict)
            assert "board" in state
            assert isinstance(state["board"], list)
            
            # === 第四步：确保能收到每一步棋 ===
            # 等待至少一步有效的落子
            move_received = False
            for _ in range(10):  # 最多等待10步
                response = websocket.receive_json()
                if response["type"] == "game_state":
                    move_received = True
                    # 验证游戏状态数据的格式
                    state = response["state"]
                    assert isinstance(state, dict)
                    assert "board" in state
                    assert isinstance(state["board"], list)
                    assert "current_player" in state
                    assert "game_over" in state
                    assert "last_move" in state
                    break
                
            assert move_received, "没有收到任何有效的落子"
            
        except Exception as e:
            pytest.fail(f"WebSocket测试失败: {str(e)}") 