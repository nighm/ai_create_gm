# 贡献指南

👋 欢迎来到项目贡献指南！

## 📝 贡献流程

1. 🍴 Fork 本仓库
2. 🌿 创建特性分支
3. ✍️ 提交更改
4. 🔄 同步上游更改
5. 📬 创建 Pull Request

## 🎯 贡献类型

### 💻 代码贡献

#### AI模型开发
- 优化神经网络架构
- 改进训练算法
- 提供新的评估方法
- 实现新的AI策略

#### 游戏功能
- 添加新的游戏模式
- 实现新的界面功能
- 优化性能
- 改进用户体验

#### 系统功能
- 实现新功能
- 修复bug
- 优化性能
- 改进安全性

### 📚 文档贡献

- 完善使用文档
- 添加示例代码
- 更新API文档
- 翻译文档

### ⚡ 性能优化

- 优化代码效率
- 减少资源消耗
- 改进响应速度
- 优化内存使用

### 🎨 界面改进

- 优化用户界面
- 改进交互体验
- 添加新的主题
- 实现响应式设计

## 💡 开发指南

### 🔧 环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/python-project-template.git
cd python-project-template
```

2. 安装依赖：
```bash
poetry install
```

3. 激活环境：
```bash
poetry shell
```

### 🏗️ 开发流程

1. 创建分支：
```bash
git checkout -b feature/your-feature
```

2. 开发功能

3. 运行测试：
```bash
poetry run pytest
```

4. 提交更改：
```bash
git add .
git commit -m "feat: add new feature"
```

### 📋 提交规范

#### 分支命名

- 功能开发：`feature/功能名称`
- 问题修复：`fix/问题描述`
- 文档更新：`docs/更新内容`
- AI相关：`ai/改进描述`

#### 提交信息

```
<类型>(<范围>): <描述>

[可选的详细描述]

[可选的脚注]
```

类型包括：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `build`: 构建相关
- `ci`: CI配置
- `chore`: 其他更改
- `ai`: AI模型相关

示例：
```
feat(ai): 添加新的神经网络层

- 实现注意力机制
- 添加残差连接
- 优化激活函数

Closes #123
```

### 🔍 代码审查

提交PR前请确保：

1. ✅ 所有测试通过
2. 📝 更新相关文档
3. 🎯 代码符合规范
4. 🔍 无安全问题
5. 📊 性能达标

## 🎮 游戏开发指南

### AI模型开发

1. 模型架构
```python
class GomokuAI(nn.Module):
    def __init__(self):
        super().__init__()
        # 实现模型结构
```

2. 训练流程
```python
def train_model(model, data):
    # 实现训练逻辑
```

### 游戏功能开发

1. 添加新模式
```python
@app.get("/game/{mode}")
async def start_game(mode: str):
    # 实现游戏模式
```

2. 实现新功能
```python
class GameFeature:
    def __init__(self):
        # 实现功能逻辑
```

## ❓ 常见问题

### Q: 如何运行测试？
```bash
poetry run pytest
```

### Q: 如何生成文档？
```bash
poetry run sphinx-build -b html docs/source docs/build
```

### Q: 如何提交PR？
1. Fork仓库
2. 创建分支
3. 提交更改
4. 创建PR

## 📞 联系方式

- 📧 Email: your.email@example.com
- 💬 Discord: [加入我们](https://discord.gg/your-server)
- 📢 Issues: [提交问题](https://github.com/yourusername/python-project-template/issues)

## 📜 行为准则

请参阅 [行为准则](CODE_OF_CONDUCT.md)。

---

感谢您的贡献！🎉
