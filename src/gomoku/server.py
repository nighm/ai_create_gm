from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Dict, Set
import os
from pathlib import Path

from .game import GomokuGame
from .ai_model import GomokuAI

app = FastAPI()

# 挂载静态文件目录
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# 存储活动的WebSocket连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.games: Dict[str, GomokuGame] = {}
        self.ai_players: Dict[str, GomokuAI] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.games:
            del self.games[client_id]
        if client_id in self.ai_players:
            del self.ai_players[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

@app.get("/")
async def get():
    with open(static_path / "index.html", encoding='utf-8') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_game":
                # 创建新游戏
                game_mode = message.get("mode", "human_vs_ai")
                game = GomokuGame()
                manager.games[client_id] = game
                
                if game_mode in ["human_vs_ai", "ai_vs_ai"]:
                    ai = GomokuAI()
                    # 如果存在预训练模型，加载它
                    model_path = static_path / "ai_model.pth"
                    if model_path.exists():
                        ai.load_model(str(model_path))
                    manager.ai_players[client_id] = ai
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "game_state",
                        "state": game.get_state()
                    }),
                    client_id
                )
                
                # 如果是AI对战模式，开始自动对战
                if game_mode == "ai_vs_ai":
                    asyncio.create_task(ai_vs_ai_game(client_id))
            
            elif message["type"] == "move":
                game = manager.games.get(client_id)
                if game and not game.game_over:
                    row, col = message["row"], message["col"]
                    if game.make_move(row, col):
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "game_state",
                                "state": game.get_state()
                            }),
                            client_id
                        )
                        
                        # 如果是人机对战模式，AI走棋
                        if not game.game_over and client_id in manager.ai_players:
                            ai = manager.ai_players[client_id]
                            ai_row, ai_col = ai.get_move(game, training=False)
                            if game.make_move(ai_row, ai_col):
                                await manager.send_personal_message(
                                    json.dumps({
                                        "type": "game_state",
                                        "state": game.get_state()
                                    }),
                                    client_id
                                )
            
            elif message["type"] == "reset":
                if client_id in manager.games:
                    manager.games[client_id].reset()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "game_state",
                            "state": manager.games[client_id].get_state()
                        }),
                        client_id
                    )

    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def ai_vs_ai_game(client_id: str):
    """
    AI之间的对战功能
    
    这就像是让AI自己和自己下棋：
    1. AI分析局面
    2. 决定下一步
    3. 在棋盘上落子
    4. 从结果中学习
    
    整个过程就像是AI在"自我练习"
    """
    # 获取这局游戏需要的东西
    game = manager.games.get(client_id)  # 获取棋盘
    ai = manager.ai_players.get(client_id)  # 获取AI选手
    
    # 如果没有找到游戏或AI，就退出
    if not game or not ai:
        return
    
    # === 持续对战主循环 ===
    while True:  # 添加外层循环，使AI持续对战
        # === AI对战主循环 ===
        # AI会不断下棋，直到分出胜负
        while not game.game_over:
            # 等待0.05秒，实现闪电般的下棋速度
            await asyncio.sleep(0.05)
            
            # === 第一步：AI思考 ===
            # AI决定下一步棋下在哪里
            row, col = ai.get_move(game, training=True)
            
            # === 第二步：落子 ===
            # 在棋盘上放下棋子
            if game.make_move(row, col):
                # === 第三步：通知观众 ===
                # 把最新的棋盘状态展示给观看的人
                await manager.send_personal_message(
                    json.dumps({
                        "type": "game_state",
                        "state": game.get_state()
                    }),
                    client_id
                )
        
        # === 第四步：学习 ===
        # 游戏结束了，AI要从这局游戏中学习
        if game.game_over:
            # 计算这局棋的得分：
            # - 赢了得1分（reward = 1.0）
            # - 输了扣1分（reward = -1.0）
            reward = 1.0 if game.winner == game.current_player else -1.0
            
            # 记录这一步的经验，包括：
            # 1. 当前的局面（state）
            state = ai.get_state_tensor(game.board, game.current_player)
            
            # 2. 选择的位置（action）
            # 把二维位置(row, col)转换为一维数字
            action = row * game.board_size + col
            
            # 3. 下完这步棋后的局面（next_state）
            next_state = ai.get_state_tensor(game.board, 3 - game.current_player)
            
            # 把这步棋的经验存起来
            ai.store_transition(state, action, reward, next_state, True)
            
            # AI开始学习（就像人类复盘总结）
            ai.train()
            
            # 保存AI学到的经验（把"大脑"保存下来）
            ai.save_model(str(static_path / "ai_model.pth"))
            
            # === 第五步：开始新的对局 ===
            # 重置游戏
            game.reset()
            # 通知前端重置了游戏状态
            await manager.send_personal_message(
                json.dumps({
                    "type": "game_state",
                    "state": game.get_state()
                }),
                client_id
            ) 