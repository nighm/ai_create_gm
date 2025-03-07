import torch  # PyTorch是一个用于深度学习的Python库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
import numpy as np  # 用于数值计算的库
from typing import List, Tuple  # 类型提示
import random  # 随机数生成
from collections import deque  # 双端队列，用于存储游戏经验

class GomokuNet(nn.Module):
    """
    五子棋AI的神经网络模型
    这就像是AI的"大脑"，用来学习和决策下棋
    """
    def __init__(self, board_size: int = 15):
        """
        初始化AI的"大脑"结构
        board_size: 棋盘大小，默认15x15
        """
        super().__init__()
        self.board_size = board_size
        
        # 卷积层：用于识别棋盘上的各种模式
        # 就像人类玩家会观察棋子的排列模式一样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # 第一层：识别基本模式
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 第二层：识别更复杂的模式
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 第三层：识别高级模式
        
        # 策略头：决定在哪里下棋
        # 就像人类玩家思考"我应该在哪里下这一步"
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头：评估当前局面的好坏
        # 就像人类玩家评估"现在是谁占优势"
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        神经网络的前向传播，即AI思考的过程
        x: 输入的棋盘状态
        """
        # 1. 分析棋盘特征
        x = F.relu(self.conv1(x))  # 第一层分析
        x = F.relu(self.conv2(x))  # 第二层分析
        x = F.relu(self.conv3(x))  # 第三层分析
        
        # 2. 决定下棋位置（策略）
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # 转换为概率分布
        
        # 3. 评估局面（价值）
        value = F.relu(self.value_conv(x))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # 输出-1到1之间的分数
        
        return policy, value

class GomokuAI:
    """
    五子棋AI玩家
    包含了AI的决策和学习过程
    """
    def __init__(self, board_size: int = 15):
        """
        初始化AI玩家
        """
        self.board_size = board_size
        # 选择使用GPU还是CPU进行计算
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 创建AI的"大脑"
        self.model = GomokuNet(board_size).to(self.device)
        # 设置学习算法
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # 创建经验池，存储对弈经验
        self.memory = deque(maxlen=10000)
        # 设置训练参数
        self.batch_size = 32  # 每次学习使用的经验数量
        self.epsilon = 1.0    # 探索率：初始时完全随机
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减

    def get_state_tensor(self, board, current_player):
        """
        将棋盘转换为AI可以理解的格式
        就像把棋盘拍照片给AI看
        """
        # 创建三个"视角"的棋盘图
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 视角1：我的棋子在哪里
        state[0] = (board == current_player)
        # 视角2：对手的棋子在哪里
        state[1] = (board == 3 - current_player)
        # 视角3：哪里是空的
        state[2] = (board == 0)
        
        # 转换为PyTorch格式
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def get_move(self, game, training=True):
        """
        决定下一步棋下在哪里
        game: 当前游戏状态
        training: 是否在训练模式
        """
        # 获取当前棋盘状态
        state = self.get_state_tensor(game.board, game.current_player)
        
        # 训练模式下，有一定概率随机下棋（探索新策略）
        if training and random.random() < self.epsilon:
            valid_moves = game.get_valid_moves()
            return random.choice(valid_moves)
        
        # 使用AI模型预测最佳落子位置
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不需要计算梯度
            policy, value = self.model(state)
            policy = policy.exp().view(self.board_size, self.board_size)
        
        # 在所有合法位置中选择最佳位置
        valid_moves = game.get_valid_moves()
        move_probs = torch.zeros(len(valid_moves))
        for i, (row, col) in enumerate(valid_moves):
            move_probs[i] = policy[row, col]
        
        # 选择概率最高的位置
        move_idx = move_probs.argmax().item()
        return valid_moves[move_idx]

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储对弈经验，供后续学习使用
        就像人类玩家在复盘时记住精彩的对局
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        训练AI，从经验中学习
        就像人类玩家通过复盘提升棋力
        """
        # 经验太少就不学习
        if len(self.memory) < self.batch_size:
            return
        
        # 随机选择一些经验进行学习
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 将经验数据转换为张量格式
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 开始学习
        self.model.train()
        self.optimizer.zero_grad()
        
        # 计算当前状态的策略和价值
        current_policy, current_value = self.model(states)
        
        # 计算目标价值
        with torch.no_grad():
            _, next_value = self.model(next_states)
            target_value = rewards + (1 - dones) * 0.99 * next_value
        
        # 计算损失（衡量预测与实际的差距）
        value_loss = F.mse_loss(current_value, target_value)  # 价值损失
        policy_loss = F.nll_loss(current_policy, actions)     # 策略损失
        total_loss = value_loss + policy_loss                 # 总损失
        
        # 更新模型参数
        total_loss.backward()
        self.optimizer.step()
        
        # 降低探索率（随着学习逐渐减少随机性）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        """
        保存AI模型
        就像保存AI的"大脑"状态
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        """
        加载已保存的AI模型
        就像恢复AI的"大脑"状态
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 