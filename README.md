# Python项目模板

[![CI Status](https://github.com/nighm/python-project-template/workflows/CI/badge.svg)](https://github.com/nighm/python-project-template/actions)
[![Documentation Status](https://readthedocs.org/projects/python-project-template/badge/?version=latest)](https://python-project-template.readthedocs.io/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/poetry-package%20manager-blue)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Changelog](https://img.shields.io/badge/changelog-查看-blue)](CHANGELOG.md)

一个专业的Python项目模板，专注于提供企业级应用开发所需的完整项目结构和最佳实践指南。本模板集成了现代Python开发中常用的工具链和框架，包括Poetry依赖管理、FastAPI Web框架、SQLAlchemy ORM、Celery任务队列等，并配备了完整的测试框架和文档生成工具，帮助开发团队快速构建高质量的Python应用。

> 📝 查看[更新日志](CHANGELOG.md)了解最新变更。

## ✨ 特性

- 📁 标准化的项目结构，包含核心模块、工具类、配置管理等
- 🌍 基于Babel的国际化支持，支持多语言翻译和本地化
- 📚 使用Sphinx自动生成API文档，支持中英双语
- ✅ 集成pytest测试框架，支持单元测试、集成测试和性能测试
- 📊 基于Prometheus的监控系统，包含健康检查和性能指标收集
- ⚡ 基于asyncio的异步编程支持，提高I/O密集型任务性能
- 🐳 完整的Docker化支持，包含多阶段构建和开发环境配置
- 🔄 GitHub Actions自动化CI/CD流程，支持测试、构建和部署

## 🚀 快速开始

### 使用此模板

1. 点击GitHub仓库页面上的"Use this template"按钮创建新项目
2. 克隆你的新项目到本地：
```bash
# 替换为你的项目URL
git clone https://github.com/your-username/your-project-name.git
cd your-project-name

# 初始化项目配置
cp .env.example .env
# 编辑.env文件，设置必要的环境变量
```

3. 安装依赖
```bash
# 安装Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 安装项目依赖
poetry install
```

4. 初始化项目
```bash
# 安装pre-commit钩子
poetry run pre-commit install

# 运行测试确保一切正常
poetry run pytest
```

## 📁 项目结构

```
.
├── src/                    # 源代码目录
│   ├── core/              # 核心功能模块
│   ├── utils/             # 通用工具
│   └── config/            # 配置文件
├── tests/                 # 测试目录
│   ├── unit/             # 单元测试
│   ├── integration/      # 集成测试
│   └── performance/      # 性能测试
├── docs/                  # 文档
│   ├── en/               # 英文文档
│   └── zh/               # 中文文档
├── examples/             # 示例代码
├── scripts/              # 工具脚本
└── resources/            # 资源文件
```

## 💻 开发指南

### 配置开发环境

1. 创建配置文件
```bash
cp .env.example .env
```

2. 修改配置
编辑 `.env` 文件，设置以下必要的环境变量：
```bash
# 应用配置
APP_ENV=development
APP_DEBUG=true
APP_PORT=8000

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=myapp
DB_USER=postgres
DB_PASSWORD=secret

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 运行测试

```bash
# 运行所有测试
poetry run pytest

# 运行特定测试
poetry run pytest tests/unit/

# 生成覆盖率报告
poetry run pytest --cov=src
```

### 代码质量

```bash
# 格式化代码
poetry run black src tests

# 类型检查
poetry run mypy src

# 代码风格检查
poetry run flake8 src
```

### 生成文档

```bash
# 生成英文文档
cd docs/en && poetry run make html

# 生成中文文档
cd docs/zh && poetry run make html
```

## 🚀 如何运行应用

1. 确保已安装依赖：
   ```bash
   poetry install
   ```
2. 运行应用：
   ```bash
   poetry run python -m src
   ```
3. 访问应用：
   - 如果是Web应用，打开浏览器并访问`http://localhost:8000`（或其他指定端口）。

## 🛠️ 项目定制

### 1. 修改项目信息

更新以下文件中的项目信息：
- `pyproject.toml`：项目名称、版本等
- `docs/conf.py`：文档信息
- `README.md`：项目描述

### 2. 选择需要的组件

本模板提供多个可选组件，你可以根据项目需求选择使用：

- 🗃️ 数据库支持 (SQLAlchemy)
  ```python
  from src.core.database import get_db
  
  db = next(get_db())
  ```

- 📦 缓存支持 (Redis)
  ```python
  from src.core.cache import get_cache
  
  cache = get_cache()
  await cache.set("key", "value", expire=3600)
  ```

- 📨 任务队列 (Celery)
  ```python
  from src.core.tasks import celery_app
  
  @celery_app.task
  def my_task():
      pass
  ```

- 🌐 API服务 (FastAPI)
  ```python
  from fastapi import FastAPI
  
  app = FastAPI()
  
  @app.get("/")
  async def root():
      return {"message": "Hello World"}
  ```

- 🖥️ CLI工具 (Click)
  ```python
  import click
  
  @click.command()
  def cli():
      click.echo("Hello World")
  ```

根据需要保留或删除相关代码。

### 3. 配置CI/CD

根据你的需求修改 `.github/workflows/` 中的工作流配置。

## 📝 最佳实践

- 使用 Poetry 管理依赖
- 编写详细的文档字符串
- 添加类型注解
- 保持较高的测试覆盖率
- 使用异步编程处理I/O操作
- 实现健康检查和监控
- **错误处理**：确保在代码中添加适当的错误处理逻辑
- **日志记录**：使用`structlog`或`logging`模块进行日志记录

## ❓ 常见问题

### 如何安装依赖？
使用以下命令安装项目的所有依赖：
```bash
poetry install
```

### 如何运行测试？
使用以下命令运行所有测试：
```bash
poetry run pytest
```

### 如何生成文档？
使用以下命令生成英文和中文文档：
```bash
cd docs/en && poetry run make html
cd docs/zh && poetry run make html
```

## 📋 路线图

- [ ] 添加更多组件示例
- [ ] 完善性能测试框架
- [ ] 增加更多部署选项
- [ ] 优化开发工具链

## 📄 许可证

[MIT License](LICENSE)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 👥 维护者

- [@nighm](https://github.com/nighm)

## 🌟 致谢

感谢所有为这个项目做出贡献的开发者！

# AI五子棋游戏

这是一个基于深度强化学习的五子棋AI游戏项目。游戏支持人机对战和AI自我对战模式，AI模型具有自学习能力，可以通过对战不断提升棋力。

## 功能特点

- 人机对战：玩家可以与AI进行对战
- AI对战：两个AI模型可以进行自动对战，并通过对战结果进行学习
- 实时学习：AI模型会在每局游戏结束后更新参数，不断提升棋力
- 模型持久化：训练后的模型会自动保存，下次启动时可以继续使用
- 美观的界面：使用HTML5 Canvas绘制棋盘，支持响应式设计

## 技术栈

- 后端：Python, FastAPI, WebSocket
- AI：PyTorch, NumPy
- 前端：HTML5, JavaScript, Canvas

## 安装和运行

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

2. 安装依赖：

```bash
pip install poetry
poetry install
```

3. 运行游戏：

```bash
python src/main.py
```

4. 打开浏览器访问：

```
http://127.0.0.1:8000
```

## 游戏说明

1. 点击"人机对战"按钮开始与AI对战，玩家执黑先手
2. 点击"AI对战"按钮观看两个AI进行对战并学习
3. 点击"重新开始"按钮可以随时重置游戏
4. 游戏结束后会显示胜负结果

## AI模型说明

- 使用深度卷积神经网络作为策略网络和价值网络
- 采用深度强化学习方法进行训练
- 使用经验回放提高训练效率
- 具有探索与利用的平衡机制

## 项目结构

```
src/
  ├── gomoku/
  │   ├── __init__.py
  │   ├── game.py          # 游戏核心逻辑
  │   ├── ai_model.py      # AI模型实现
  │   ├── server.py        # Web服务器和API
  │   └── static/          # 静态文件
  │       └── index.html   # 游戏界面
  └── main.py              # 程序入口
```

## 开发计划

- [ ] 添加更多的AI算法实现
- [ ] 支持调整AI难度
- [ ] 添加游戏记录回放功能
- [ ] 支持多人在线对战
- [ ] 优化AI训练效率

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

# Gomoku AI

这是一个基于深度强化学习的五子棋AI项目。AI通过自我对弈不断学习和提升，能够在网页界面上与人类玩家进行对战。

## 功能特点

- 基于深度强化学习的AI模型
- 实时网页对战界面
- 支持人机对战和AI自我对弈
- AI持续学习并保存模型

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/[your-username]/ai_create_gm.git
cd ai_create_gm
```

2. 安装依赖：
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
```

3. 运行游戏：
```bash
python run.py
```

4. 在浏览器中打开：
```
http://localhost:8000
```

## 项目结构

```
ai_create_gm/
├── src/
│   ├── gomoku/
│   │   ├── ai_model.py    # AI模型实现
│   │   ├── game.py        # 游戏核心逻辑
│   │   └── server.py      # Web服务器
│   └── main.py            # 主程序入口
├── tests/                 # 测试文件
└── run.py                 # 运行脚本
```

## 开发

运行测试：
```bash
python -m pytest tests/
```

## 许可证

MIT License
