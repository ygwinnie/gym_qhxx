import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _in_streamlit():
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if not _in_streamlit():
    print("Run this app with: streamlit run Home.py")
else:
    st.set_page_config(
        page_title="RL Lab - 强化学习实验室",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 强化学习实验室 (RL Lab)")
    st.subheader("Grade 8 Artificial Intelligence Course")

    st.markdown("""
    欢迎来到 **强化学习实验室**！在这里，我们将通过一系列有趣的实验，探索人工智能是如何通过“试错”来学习的。

    ---

    ### 📚 课程目录

    #### [1. 基础篇：冰湖探险 (FrozenLake)](/FrozenLake)
    *   **任务**: 训练一个小精灵在冰面上行走，避开冰窟窿，拿到礼物。
    *   **核心概念**: 
        *   状态 (State) 与 动作 (Action)
        *   Q表格 (Q-Table)
        *   探索与利用 (Exploration vs Exploitation)

    #### [2. 进阶篇：月球着陆 (LunarLander)](/LunarLander)
    *   **任务**: 控制登月舱平稳着陆在月球表面。
    *   **核心概念**:
        *   连续状态空间
        *   物理模拟
        *   深度强化学习 (DQN)

    ---

    ### 💡 如何使用
    请点击左侧边栏的页面名称，切换不同的实验项目。

    *Developed for Grade 8 AI Curriculum.*
    """)
