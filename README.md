# 🤖 强化学习实验室 (Reinforcement Learning Lab)

这是一个专为 **8年级学生** 设计的互动教学工具，旨在通过可视化的方式，帮助学生理解 **强化学习 (Reinforcement Learning)** 的核心概念。

项目包含两个核心实验章节：
1.  **基础篇：冰湖探险 (FrozenLake)** - 学习离散状态下的 Q-Learning。
2.  **进阶篇：月球着陆 (LunarLander)** - 探索连续状态空间与深度强化学习 (DQN) 的威力。

## ✨ 主要功能

### 1. 冰湖探险 (FrozenLake)
*   **可视化参数调整**: 实时调整学习率、探索率等参数。
*   **多维度数据分析**: 胜率、步数、奖励、探索率曲线。
*   **策略可视化**: 策略地图与 Q 表热力图。
*   **实战演示**: 10 轮连续动画演示，验证训练成果。

### 2. 月球着陆 (LunarLander)
*   **连续状态挑战**: 体验传统 Q-Learning 在处理连续状态（位置、速度、角度）时的局限性（维度灾难）。
*   **深度强化学习**: 演示 DQN 如何利用神经网络成功解决复杂控制问题。
*   **对比实验**: 直观对比表格型 Q-Learning 与 DQN 的表现差异。

## 🚀 快速开始 (Quick Start)

### 1. 环境要求
*   Python 3.8 或更高版本
*   建议使用 Anaconda 或 Virtualenv 创建独立环境

### 2. 安装依赖
在终端中运行以下命令安装所需库：

```bash
pip install -r requirements.txt
```

### 3. 运行程序
安装完成后，启动 Streamlit 应用：

```bash
streamlit run Home.py
```

程序启动后，浏览器会自动打开 `http://localhost:8501`。

## 📂 文件结构

*   `Home.py`: **课程主页**。项目的入口页面。
*   `pages/`: **课程章节**
    *   `1_🧊_FrozenLake.py`: 冰湖探险课程（Q-Learning 基础）。
    *   `2_🚀_LunarLander.py`: 月球着陆课程（连续空间与 DQN）。
*   `requirements.txt`: **依赖列表**。列出了项目运行所需的 Python 库。
*   `docs/`: **文档目录**
    *   `ARCHITECTURE.md`: **技术架构文档**。详细说明了系统的核心模块设计、状态管理机制和数据字典。
    *   `USER_MANUAL.md`: **用户手册**。包含参数详解和常见问题解答。
    *   `WORKSHEET.md`: **学生任务单**。包含探究性学习任务。

## 📖 教学建议

1.  **循序渐进**: 先完成 FrozenLake，理解 Q 表和奖励机制，再进入 LunarLander。
2.  **对比分析**: 在 LunarLander 中，重点引导学生观察为什么 Q-Learning 会失败（状态太多，学不过来），从而引出 DQN 的必要性。
3.  **参数探究**: 鼓励学生大胆调整参数，比如把“坠毁惩罚”调得非常大，看看机器人的行为会有什么变化。

---
*Developed for Grade 8 AI Curriculum.*
