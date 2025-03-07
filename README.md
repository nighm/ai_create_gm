# AI五子棋项目

[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/nighm/ai_create_gm/releases/tag/v1.1.0)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

基于深度强化学习的五子棋AI项目，提供完整的项目结构和开发框架。本项目使用现代Python技术栈，包括FastAPI用于Web服务、PyTorch用于AI模型开发、Redis用于缓存、SQLAlchemy用于数据持久化等。

> 📝 查看[更新日志](CHANGELOG.md)了解最新变更。

## ✨ 特性

- 🤖 深度强化学习模型框架
  - 预配置的神经网络结构
  - 完整的训练流程
  - 模型评估和验证
- 🎮 游戏系统架构
  - 五子棋核心逻辑
  - AI对弈系统
  - 实时对战功能
- 🗃️ 数据管理
  - 游戏记录存储
  - 模型参数管理
  - 用户数据处理
- 📦 缓存系统
  - 游戏状态缓存
  - AI模型缓存
  - 会话管理
- 📨 异步任务
  - AI训练任务
  - 数据分析任务
  - 系统维护任务
- 🌐 Web服务
  - RESTful API
  - WebSocket实时通信
  - 交互式游戏界面
- 🔒 安全特性
  - 用户认证
  - 请求验证
  - 数据加密
- 📊 监控和日志
  - 性能监控
  - 错误追踪
  - 系统日志

## 🚀 快速开始

### 环境准备

1. 克隆项目：
   ```bash
   git clone https://github.com/nighm/ai_create_gm.git
   cd ai_create_gm
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. 安装依赖：
   ```bash
   pip install -e .
   ```

### 运行项目

1. 启动服务：
   ```bash
   python run.py
   ```

2. 访问应用：
   - Web界面：`http://localhost:8000`
   - API文档：`http://localhost:8000/docs`
   - 管理界面：`http://localhost:8000/admin`

## 🛠️ 项目定制

### 1. 修改项目信息

更新以下文件中的项目信息：
- `pyproject.toml`：AI模型配置、依赖版本等
- `docs/conf.py`：文档信息
- `README.md`：项目描述

### 2. 选择需要的组件

根据五子棋AI项目需求选择使用：

- 🗃️ 数据库支持 (SQLAlchemy)
  ```python
  from src.core.database import get_db
  
  # 存储游戏记录
  db = next(get_db())
  ```

- 📦 缓存支持 (Redis)
  ```python
  from src.core.cache import get_cache
  
  # 缓存AI模型状态
  cache = get_cache()
  await cache.set("model_state", state, expire=3600)
  ```

- 📨 任务队列 (Celery)
  ```python
  from src.core.tasks import celery_app
  
  @celery_app.task
  def train_ai_model():
      # AI模型训练任务
      pass
  ```

- 🌐 API服务 (FastAPI)
  ```python
  from fastapi import FastAPI
  
  app = FastAPI()
  
  @app.post("/move")
  async def make_move(position: dict):
      return {"next_move": ai.predict(position)}
  ```

### 3. 配置CI/CD

根据需求修改 `.github/workflows/` 中的工作流配置，包括：
- AI模型测试
- 性能测试
- 部署流程

## 📝 最佳实践

- 使用 Poetry 管理依赖
- 编写详细的文档字符串
- 添加类型注解
- 保持较高的测试覆盖率
- 使用异步编程处理I/O操作
- 实现健康检查和监控
- **错误处理**：实现完整的游戏异常处理
- **日志记录**：记录AI决策过程和游戏状态

## ❓ 常见问题

### 如何训练AI模型？
使用以下命令启动模型训练：
```bash
python -m src.train
```

### 如何运行测试？
运行所有测试套件：
```bash
pytest
```

### 如何生成文档？
生成项目文档：
```bash
cd docs && make html
```

## 📋 路线图

- [ ] 优化AI模型架构
- [ ] 添加更多游戏模式
- [ ] 实现联机对战
- [ ] 增加AI评估工具
- [ ] 优化开发工具链

## 📄 许可证

[MIT License](LICENSE)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 👥 维护者

- [@nighm](https://github.com/nighm)

## 🌟 致谢

感谢所有为这个项目做出贡献的开发者！
