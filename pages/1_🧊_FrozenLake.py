import streamlit as st
import gymnasium as gym
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面标题和布局
st.set_page_config(page_title="FrozenLake 强化学习实验室", layout="wide")

st.title("🤖 强化学习实验室: FrozenLake")
st.markdown("""
欢迎来到强化学习实验室！在这里，你将扮演一名**AI 训练师**。
你的任务是调整参数，训练一个小精灵(Elf)学会安全地穿过冰湖拿到礼物。
""")

# ==========================================
# 侧边栏: 参数控制台
# ==========================================
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .st-emotion-cache-16txtl3 {
            padding-top: 1rem;
        }
        /* 垂直居中对齐侧边栏的列 */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        /* 极致紧凑模式：减少组件间的垂直间距 */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.2rem;
        }
        /* 微调文字和滑块的边距 */
        [data-testid="stSidebar"] .stMarkdown {
            margin-bottom: -5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ 控制台")
    
    # 辅助函数：紧凑型滑块 (标签在左，滑块在右)
    def compact_slider(label, min_v, max_v, default_v, step=None, format=None, help=None):
        col1, col2 = st.columns([0.35, 0.65]) # 左侧文字占 35%，右侧滑块占 65%
        with col1:
            # 去掉 div wrapper，直接使用 markdown，让问号图标能自然跟随在文字后面
            st.markdown(f"**{label}**", help=help)
        with col2:
            return st.slider("", min_v, max_v, default_v, step=step, format=format, label_visibility="collapsed")

    # 1. 基础设置 (常驻)
    episodes = compact_slider("训练轮数", 100, 5000, 2000, help="机器人练习的次数。次数越多，它学得越好，但花的时间也越长。")
    is_slippery = st.checkbox("冰面打滑 (Slippery)", value=True, help="如果选中，冰面会很滑！机器人想往左走，可能会滑到上面或下面。这增加了难度。")
    
    # 2. 算法参数
    st.markdown("##### 🧠 算法参数")
    learning_rate = compact_slider("学习率", 0.01, 1.0, 0.8, help="机器人接受新知识的速度。太高容易‘喜新厌旧’（不稳定），太低则‘固步自封’（学得慢）。")
    discount_factor = compact_slider("折扣因子", 0.1, 1.0, 0.95, help="机器人有多看重未来的奖励。0表示‘只看眼前’（短视），1表示‘高瞻远瞩’（重视长期利益）。")
    
    st.caption("探索策略 (Epsilon)")
    epsilon_start = compact_slider("初始探索", 0.1, 1.0, 1.0, help="刚开始时，机器人有多大几率‘瞎逛’（尝试新路线）。1.0 表示完全在瞎逛。")
    epsilon_decay = compact_slider("探索衰减", 0.90, 0.9999, 0.995, format="%.4f", help="随着时间推移，机器人减少‘瞎逛’的速度。数值越小，它‘收心’得越快，越早开始利用学到的经验。")
    min_epsilon = compact_slider("最小探索", 0.0, 0.5, 0.01, help="即使学得差不多了，机器人也会保留一点点好奇心（瞎逛的几率），防止错过更好的路。")

    # 3. 高级设置
    st.markdown("##### ⚙️ 高级设置")
    hole_penalty = compact_slider("掉坑惩罚", -10.0, 0.0, 0.0, step=0.5, help="掉进冰窟窿的惩罚分数。惩罚越重（负分越多），它越害怕掉进去。")
    step_penalty = compact_slider("步数消耗", -1.0, 0.0, 0.0, step=0.01, help="每走一步的体力消耗。如果每一步都扣分，它会想办法尽快跑到终点。")

    st.markdown("---")
    start_btn = st.button("🚀 开始训练", type="primary", use_container_width=True)

# ==========================================
# 状态管理 (Session State)
# ==========================================
if 'trained_q_table' not in st.session_state:
    st.session_state.trained_q_table = None
if 'success_history' not in st.session_state:
    st.session_state.success_history = []
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False

# ==========================================
# 核心逻辑: Q-Learning 训练
# ==========================================
def train_agent():
    # 创建环境
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=None)
    
    # 初始化 Q 表
    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = np.zeros((state_space, action_space))
    
    # 记录训练数据
    rewards_history = [] # 原始环境奖励 (0或1)
    custom_rewards_history = [] # 自定义奖励 (包含惩罚)
    steps_history = [] # 每轮步数
    epsilon_history = [] # 探索率变化
    
    # 用于绘图的聚合数据
    plot_data = {
        "episode": [],
        "success_rate": [],
        "avg_steps": [],
        "avg_custom_reward": [],
        "epsilon": []
    }
    
    epsilon = epsilon_start
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    # 定义数据记录点 (每 5% 记录一次，或者至少记录 20 个点)
    record_interval = max(1, episodes // 20)
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0 # 原始奖励
        total_custom_reward = 0 # 自定义奖励
        steps = 0
        
        while not done:
            # Epsilon-Greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # 智能打破平局 (Random Tie-Breaking)
                max_q = np.max(q_table[state, :])
                actions_with_max_q = np.where(q_table[state, :] == max_q)[0]
                action = np.random.choice(actions_with_max_q)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # --- 自定义奖励逻辑 (Reward Shaping) ---
            custom_reward = reward
            if terminated and reward == 0: # 掉坑里了
                custom_reward = hole_penalty
            elif not done: # 还在走
                custom_reward = step_penalty
            elif reward == 1: # 到达终点
                custom_reward = 1.0
            
            # 更新 Q 表
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            new_value = old_value + learning_rate * (custom_reward + discount_factor * next_max - old_value)
            q_table[state, action] = new_value
            
            state = next_state
            total_reward += reward
            total_custom_reward += custom_reward
            
        # 衰减 Epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        rewards_history.append(total_reward)
        custom_rewards_history.append(total_custom_reward)
        steps_history.append(steps)
        epsilon_history.append(epsilon)
        
        # 更新进度和记录数据
        if (episode + 1) % record_interval == 0:
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            
            # 计算最近 record_interval 轮的统计数据
            recent_rewards = rewards_history[-record_interval:]
            recent_custom_rewards = custom_rewards_history[-record_interval:]
            recent_steps = steps_history[-record_interval:]
            
            success_rate = sum(recent_rewards) / len(recent_rewards) * 100
            avg_custom_reward = sum(recent_custom_rewards) / len(recent_custom_rewards)
            avg_steps = sum(recent_steps) / len(recent_steps)
            
            plot_data["episode"].append(episode + 1)
            plot_data["success_rate"].append(success_rate)
            plot_data["avg_steps"].append(avg_steps)
            plot_data["avg_custom_reward"].append(avg_custom_reward)
            plot_data["epsilon"].append(epsilon)
            
            status_text.text(f"Training... Episode {episode+1}/{episodes} | 胜率: {success_rate:.1f}% | Epsilon: {epsilon:.4f}")

    end_time = time.time()
    st.success(f"✅ 训练完成！耗时: {end_time - start_time:.2f} 秒")
    
    # 保存到 Session State
    st.session_state.trained_q_table = q_table
    st.session_state.training_results = pd.DataFrame(plot_data).set_index("episode")
    st.session_state.training_completed = True

# ==========================================
# 结果展示
# ==========================================
if start_btn:
    train_agent()

# 只要训练过，就显示结果 (即使点击其他按钮刷新了页面)
if st.session_state.training_completed:
    q_table = st.session_state.trained_q_table
    results_df = st.session_state.training_results
    
    # --- 1. 学习曲线 (全宽) ---
    st.markdown("### 📈 学习过程分析")
    if not results_df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 胜率", "👣 平均步数", "💰 奖励分数", "🎲 探索率"])
        
        with tab1:
            st.line_chart(results_df["success_rate"], height=250)
            st.caption("胜率越高，说明机器人越容易拿到礼物。")
        with tab2:
            st.line_chart(results_df["avg_steps"], height=250)
            st.caption("步数越少，说明机器人跑得越快（或者掉坑掉得越快...结合胜率看）。")
        with tab3:
            st.line_chart(results_df["avg_custom_reward"], height=250)
            st.caption("分数越高，说明机器人越符合你的期望（少掉坑、少绕路）。")
        with tab4:
            st.line_chart(results_df["epsilon"], height=250)
            st.caption("探索率越低，机器人越依赖经验，不再瞎逛。")
    else:
        st.warning("训练轮数太少，无法生成曲线。")

    st.divider()

    # --- 2. 策略地图 & Q表数值 (并排) ---
    col_map, col_q = st.columns([1, 1])
    
    with col_map:
        st.markdown("### 🗺️ 策略地图")
        arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        grid_html = "<table style='border-collapse: collapse; margin: 0 auto;'>"
        for i in range(4):
            grid_html += "<tr>"
            for j in range(4):
                state = i * 4 + j
                desc = ["S", "F", "F", "F", "F", "H", "F", "H", "F", "F", "F", "H", "H", "F", "F", "G"]
                cell_type = desc[state]
                
                bg_color = "#f0f2f6"
                content = ""
                
                if cell_type == "H":
                    bg_color = "#ffcccb"
                    content = "🕳️"
                elif cell_type == "G":
                    bg_color = "#90ee90"
                    content = "🎁"
                elif cell_type == "S":
                    bg_color = "#add8e6"
                    content = "🏠"
                else:
                    if np.max(q_table[state, :]) == 0 and cell_type != "G":
                        content = "?"
                    else:
                        best_action = np.argmax(q_table[state, :])
                        content = f"<span style='font-size: 20px; font-weight: bold;'>{arrows[best_action]}</span>"
                
                grid_html += f"<td style='width: 50px; height: 50px; text-align: center; background-color: {bg_color}; border: 2px solid white;'>{content}</td>"
            grid_html += "</tr>"
        grid_html += "</table>"
        st.markdown(grid_html, unsafe_allow_html=True)

    with col_q:
        st.markdown("### 🧠 Q表数值")
        # 使用 empty 容器显式渲染，并加上 key 防止重绘问题
        q_table_placeholder = st.empty()
        df_q = pd.DataFrame(q_table, columns=["←", "↓", "→", "↑"])
        q_table_placeholder.dataframe(
            df_q.style.background_gradient(cmap="Blues", axis=None), 
            height=300, 
            key="q_table_display"
        )

    st.divider()

    # --- 3. 实战演示 (底部，全宽) ---
    st.markdown("### 🎥 实战演示 (10轮)")
    run_test_btn = st.button("开始测试 (Run Test)", use_container_width=True)
    
    if run_test_btn:
        # 动态创建占位符，确保只在点击后出现
        # 这里不再嵌套在其他列中，而是直接使用主布局
        # 调整比例 [1, 2] 让图片更小一些 (之前是 [1.5, 1])
        sub_c1, sub_c2 = st.columns([1, 2])
        with sub_c1:
            image_placeholder = st.empty()
        with sub_c2:
            st.markdown("#### 测试统计")
            metric_success = st.empty()
            metric_steps = st.empty()
        
        # 准备环境
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode="rgb_array")
        success_count = 0
        total_steps = 0
        
        for i in range(10):
            state, _ = env.reset()
            done = False
            steps = 0
            
            # 每一轮的动画
            while not done and steps < 50:
                frame = env.render()
                image_placeholder.image(frame, caption=f"Episode {i+1}/10 | Step {steps}", use_container_width=True)
                
                action = np.argmax(q_table[state, :])
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                time.sleep(0.1)
            
            # 显示该轮结果
            frame = env.render()
            result_msg = "Success! 🎁" if reward == 1 else "Failed ☠️"
            image_placeholder.image(frame, caption=f"Ep {i+1} Finished: {result_msg}", use_container_width=True)
            
            if reward == 1: success_count += 1
            total_steps += steps
            
            # 实时更新统计
            metric_success.metric("当前成功", f"{success_count} / {i+1}")
            metric_steps.metric("累计步数", f"{total_steps}")
            
            time.sleep(0.5)
        
        env.close()
        
        # 最终评价
        metric_success.metric("最终成功", f"{success_count} / 10")
        metric_steps.metric("平均步数", f"{total_steps / 10:.1f}")
        
        if success_count >= 8: st.success("🏆 表现优秀！")
        elif success_count >= 5: st.warning("😐 表现一般")
        else: st.error("💀 还需要努力")

else:
    st.info("👈 请在左侧调整参数，然后点击 '开始训练' 按钮。")
