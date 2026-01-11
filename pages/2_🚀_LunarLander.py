import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os

# Set page config
st.set_page_config(page_title="LunarLander 强化学习实验室", layout="wide")

st.title("🚀 强化学习实验室: LunarLander")
st.markdown("""
欢迎来到月球表面！在这里，我们将挑战一个更难的任务：**控制登月舱平稳着陆**。
这比冰湖探险难得多，因为状态是**连续**的（位置、速度、角度都是小数，而不是简单的格子）。
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
    st.markdown("### 🎛️ 实验控制台")
    
    experiment_type = st.radio(
        "选择实验模式",
        ["1. 传统 Q-Learning (失败案例)", "2. 深度 Q-Network (成功案例)"],
        index=0
    )
    
    st.divider()
    
    # 辅助函数：紧凑型滑块
    def compact_slider(label, min_v, max_v, default_v, step=None, format=None, help=None):
        col1, col2 = st.columns([0.35, 0.65])
        with col1:
            st.markdown(f"**{label}**", help=help)
        with col2:
            return st.slider("", min_v, max_v, default_v, step=step, format=format, label_visibility="collapsed")

    if experiment_type.startswith("1"):
        st.markdown("#### 🔧 Q-Learning 参数")
        
        # 1. 基础设置
        buckets = compact_slider("离散精度", 2, 10, 5, help="把每个连续变量切成多少份。份数越少越粗糙，份数越多状态爆炸。")
        episodes = compact_slider("训练轮数", 100, 2000, 500, help="训练次数。")
        
        # 2. 算法参数
        st.markdown("##### 🧠 算法参数")
        learning_rate = compact_slider("学习率", 0.01, 1.0, 0.1, help="机器人接受新知识的速度。")
        discount_factor = compact_slider("折扣因子", 0.1, 1.0, 0.99, help="机器人有多看重未来的奖励。")
        
        st.caption("探索策略 (Epsilon)")
        epsilon_start = compact_slider("初始探索", 0.1, 1.0, 1.0, help="刚开始时，机器人有多大几率‘瞎逛’。")
        epsilon_decay = compact_slider("探索衰减", 0.90, 0.9999, 0.995, format="%.4f", help="数值越小，它‘收心’得越快。")
        min_epsilon = compact_slider("最小探索", 0.0, 0.5, 0.01, help="保留一点点好奇心。")

        # 3. 高级设置 (Reward Shaping)
        st.markdown("##### ⚙️ 高级设置")
        crash_penalty = compact_slider("坠毁惩罚", -100.0, 0.0, -100.0, step=10.0, help="坠毁时的额外惩罚分数。")
        
        st.divider()
        start_q_btn = st.button("🚀 开始 Q-Learning 训练", type="primary", use_container_width=True)
        
    else:
        st.markdown("#### 🛠️ DQN 训练参数")
        st.info("调整参数，观察对训练速度和效果的影响。")
        
        lr = compact_slider("学习率", 0.0001, 0.005, 0.0005, step=0.0001, format="%.4f", help="机器人修正错误的幅度。太大容易震荡，太小学习太慢。")
        gamma = compact_slider("折扣因子", 0.90, 0.99, 0.99, format="%.2f", help="机器人有多看重未来的奖励。接近1表示有远见。")
        
        st.caption("探索策略 (Exploration)")
        exploration_initial = compact_slider("初始探索率", 0.5, 1.0, 1.0, format="%.2f", help="训练开始时随机探索的概率。")
        exploration_final = compact_slider("最终探索率", 0.01, 0.2, 0.01, format="%.2f", help="探索阶段结束后保留的探索概率。")
        exploration_fraction = compact_slider("探索占比", 0.2, 0.8, 0.5, format="%.2f", help="训练前期用于探索的时间比例。")
        
        batch_size = compact_slider("批次大小", 32, 256, 64, step=32, help="每次从经验池中复习多少条经验。")
        total_timesteps = compact_slider("训练步数", 10000, 200000, 100000, step=10000, help="训练的总时长。步数越多，效果越好。（100000步约需7-10分钟）")
        
        st.caption("神经网络结构")
        network_size = st.radio(
            "网络大小",
            ["简单 (128-128)", "标准 (256-256) 推荐", "复杂 (512-256)"],
            index=1,
            help="网络越大，学习能力越强，但训练越慢。标准配置适合课堂使用。"
        )
        
        st.divider()
        start_train_btn = st.button("🚀 开始训练 (Start Training)", type="primary", use_container_width=True)

# ==========================================
# Helper: Discretized Wrapper
# ==========================================
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_buckets=5):
        super().__init__(env)
        self.n_buckets = n_buckets
        
        # LunarLander-v2 state has 8 dimensions:
        # [x, y, vx, vy, angle, v_angle, left_leg, right_leg]
        # We define bounds for each to discretize them.
        # Note: These bounds are approximate.
        self.bounds = [
            (-1.0, 1.0),   # x
            (-0.5, 1.5),   # y
            (-2.0, 2.0),   # vx
            (-2.0, 2.0),   # vy
            (-1.0, 1.0),   # angle
            (-2.0, 2.0),   # v_angle
            (0.0, 1.0),    # left_leg (boolean-ish)
            (0.0, 1.0)     # right_leg (boolean-ish)
        ]
        
    def observation(self, obs):
        discretized = []
        for i, val in enumerate(obs):
            l, h = self.bounds[i]
            # Clip value to bounds
            val = min(max(val, l), h)
            # Map to bucket index
            # p is 0..1
            p = (val - l) / (h - l)
            bucket = int(p * self.n_buckets)
            bucket = min(bucket, self.n_buckets - 1)
            discretized.append(bucket)
            
        # Convert tuple of buckets to a single integer index if possible, 
        # but for Q-table we might just use the tuple as key.
        return tuple(discretized)

# ==========================================
# State Management
# ==========================================
if 'lunar_q_table' not in st.session_state:
    st.session_state.lunar_q_table = None
if 'lunar_results' not in st.session_state:
    st.session_state.lunar_results = pd.DataFrame()
if 'lunar_training_completed' not in st.session_state:
    st.session_state.lunar_training_completed = False

# ==========================================
# Part 1: Q-Learning Implementation
# ==========================================
if experiment_type.startswith("1"):
    st.subheader("🧪 实验 1: 传统 Q-Learning 的局限性")
    
    # --- Training Logic ---
    if start_q_btn:
        st.write("正在初始化环境...")
        
        # Create environment
        try:
            env = gym.make("LunarLander-v3", render_mode=None) # No render during training
            env = DiscretizedObservationWrapper(env, n_buckets=buckets)
        except Exception as e:
            st.error(f"环境创建失败: {e}")
            st.stop()
            
        # Q-Table
        q_table = {}
        
        def get_q(state, action):
            return q_table.get((state, action), 0.0)
            
        def update_q(state, action, value):
            q_table[(state, action)] = value
            
        def choose_action(state, epsilon):
            if np.random.random() < epsilon:
                return env.action_space.sample()
            else:
                q_values = [get_q(state, a) for a in range(4)]
                max_q = max(q_values)
                actions = [i for i, q in enumerate(q_values) if q == max_q]
                return np.random.choice(actions)

        # Training Loop
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        rewards_history = []
        steps_history = []
        epsilon_history = []
        
        plot_data = {
            "episode": [],
            "success_rate": [],
            "avg_steps": [],
            "avg_reward": [],
            "epsilon": []
        }
        
        epsilon = epsilon_start
        alpha = learning_rate
        gamma = discount_factor
        
        start_time = time.time()
        
        for ep in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                action = choose_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Reward Shaping
                custom_reward = reward
                if terminated and reward == -100:
                    custom_reward = crash_penalty
                
                # Update Q
                old_q = get_q(state, action)
                next_max_q = max([get_q(next_state, a) for a in range(4)])
                new_q = old_q + alpha * (custom_reward + gamma * next_max_q - old_q)
                update_q(state, action, new_q)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            rewards_history.append(total_reward)
            steps_history.append(steps)
            epsilon_history.append(epsilon)
            
            if (ep + 1) % 10 == 0:
                recent_rewards = rewards_history[-10:]
                avg_reward = np.mean(recent_rewards)
                avg_steps = np.mean(steps_history[-10:])
                success_rate = sum(r > 200 for r in recent_rewards) / len(recent_rewards) * 100
                
                plot_data["episode"].append(ep + 1)
                plot_data["success_rate"].append(success_rate)
                plot_data["avg_steps"].append(avg_steps)
                plot_data["avg_reward"].append(avg_reward)
                plot_data["epsilon"].append(epsilon)
                
                progress_bar.progress((ep + 1) / episodes)
                status_text.text(f"Training... Episode {ep+1}/{episodes} | Avg Reward: {avg_reward:.1f}")
                
        env.close()
        st.success(f"✅ 训练完成！耗时: {time.time() - start_time:.2f} 秒")
        
        # Save to session state
        st.session_state.lunar_q_table = q_table
        st.session_state.lunar_results = pd.DataFrame(plot_data).set_index("episode")
        st.session_state.lunar_training_completed = True

    # --- Results & Testing Logic ---
    if st.session_state.lunar_training_completed:
        results_df = st.session_state.lunar_results
        q_table = st.session_state.lunar_q_table
        
        # 1. Charts
        st.markdown("### 📈 学习过程分析")
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 胜率", "👣 平均步数", "💰 奖励分数", "🎲 探索率"])
        
        with tab1:
            st.line_chart(results_df["success_rate"])
            st.caption("胜率 (分数 > 200) 越高，说明着陆越成功。")
        with tab2:
            st.line_chart(results_df["avg_steps"])
            st.caption("步数越少，说明着陆越快。")
        with tab3:
            st.line_chart(results_df["avg_reward"])
            st.caption("分数越高，说明着陆质量越好。")
        with tab4:
            st.line_chart(results_df["epsilon"])
            st.caption("探索率逐渐降低，机器人越来越依赖经验。")
            
        st.info("""
        **🤔 为什么胜率这么低/不稳定？**
        
        这正是**传统 Q-Learning** 在复杂连续环境中的典型表现（失败案例）：
        1.  **状态空间爆炸**：即使我们将每个维度只切成 5 份，总状态数也高达 $5^8 = 390,625$ 个！短短几百轮训练根本无法填满 Q 表，大部分状态机器人从未见过。
        2.  **精度丢失**：为了使用 Q 表，我们将连续的位置和速度“模糊化”了（离散化）。这导致机器人无法感知细微的变化，就像戴着厚手套穿针引线，很难精准控制。
        3.  **运气成分**：偶尔出现的成功（波峰）可能只是因为初始位置较好，或者机器人“蒙”对了一条路，但它并没有真正学会通用的飞行技巧。
        
        👉 **这就是为什么我们需要深度强化学习 (DQN)！** 请尝试切换到“实验 2”看看区别。
        """)
            
        st.divider()
        
        # 2. Test Section
        st.markdown("### 🎥 实战演示 (10轮)")
        st.info("点击下方按钮，查看当前模型的实际表现。")
        run_test_btn = st.button("开始测试 (Run Test)", use_container_width=True)
        
        if run_test_btn:
            col_anim, col_stats = st.columns([2, 1])
            with col_anim:
                frame_placeholder = st.empty()
            with col_stats:
                st.markdown("#### 测试统计")
                metric_success_rate = st.empty()
                metric_avg_reward = st.empty()
                metric_steps = st.empty()
            
            # Re-create environment for rendering
            try:
                env = gym.make("LunarLander-v3", render_mode="rgb_array")
                env = DiscretizedObservationWrapper(env, n_buckets=buckets)
            except:
                st.error("环境创建失败")
                st.stop()
                
            success_count = 0
            total_steps = 0
            total_test_reward = 0
            
            # Helper to get Q
            def get_q_test(state, action):
                return q_table.get((state, action), 0.0)
            
            for i in range(10):
                state, _ = env.reset()
                done = False
                steps = 0
                episode_reward = 0
                
                while not done and steps < 500:
                    frame = env.render()
                    frame_placeholder.image(frame, caption=f"Test Episode {i+1}/10 | Step {steps}", use_container_width=True)
                    
                    # Greedy action
                    q_values = [get_q_test(state, a) for a in range(4)]
                    action = np.argmax(q_values)
                    
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    episode_reward += reward
                    time.sleep(0.01) # Slow down animation
                
                if reward == 100 or reward > 200: # Approximate success check
                    success_count += 1
                total_steps += steps
                total_test_reward += episode_reward
                
                metric_success_rate.metric("成功率 (Success Rate)", f"{success_count/(i+1)*100:.0f}% ({success_count}/{i+1})")
                metric_avg_reward.metric("平均奖励 (Avg Reward)", f"{total_test_reward/(i+1):.1f}")
                metric_steps.metric("平均步数 (Avg Steps)", f"{total_steps/(i+1):.1f}")
                
            env.close()
            
            if success_count >= 8: st.success("🏆 表现优秀！")
            elif success_count >= 5: st.warning("😐 表现一般")
            else: st.error("💀 还需要努力")

    else:
        st.info("👈 请在左侧调整参数，然后点击 '开始训练' 按钮。")

# ==========================================
# Part 2: DQN Implementation
# ==========================================
elif experiment_type.startswith("2"):
    st.subheader("🧠 实验 2: 深度强化学习 (DQN) 的威力")
    
    st.markdown("""
    **DQN (Deep Q-Network)** 使用神经网络来直接处理连续的状态输入，不再需要人工进行离散化。
    
    在这个实验室中，你将亲手训练一个神经网络！
    """)
    
    st.divider()

    # --- Main Area: Training Dashboard ---
    st.markdown("### 🏋️‍♂️ DQN 训练实验室")
    
    # Create persistent placeholders for charts and status
    status_text = st.empty()
    progress_bar = st.empty()
    
    # Three-column chart layout
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        st.markdown("**📈 平均奖励**")
        chart_reward = st.empty()
    with chart_col2:
        st.markdown("**✅ 成功率**")
        chart_success = st.empty()
    with chart_col3:
        st.markdown("**🔍 探索率**")
        chart_exploration = st.empty()
    
    # Display saved training charts if exists
    if 'dqn_training_data' in st.session_state and st.session_state.dqn_training_data is not None:
        data = st.session_state.dqn_training_data
        
        df_reward = pd.DataFrame({
            "steps": data['steps'], 
            "当前奖励": data['episode_reward'],
            "平均奖励": data['avg_reward']
        }).set_index("steps")
        chart_reward.line_chart(df_reward, height=200, color=["#1f77b4", "#cccccc"])
        
        df_success = pd.DataFrame({"steps": data['steps'], "success_rate": data['success_rate']}).set_index("steps")
        chart_success.line_chart(df_success, height=200)
        
        df_exploration = pd.DataFrame({"steps": data['steps'], "exploration_rate": data['exploration_rate']}).set_index("steps")
        chart_exploration.line_chart(df_exploration, height=200)
        
        status_text.success(f"✅ 训练完成！(共 {data['steps'][-1]} 步)")

    # --- Training Logic ---
    if start_train_btn:
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError:
            st.error("请先安装 stable-baselines3: `pip install stable-baselines3 shimmy gymnasium[box2d]`")
            st.stop()

        # Custom Callback for Streamlit
        class StreamlitCallback(BaseCallback):
            def __init__(self, status_text, progress_bar, chart_reward, chart_success, chart_exploration, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []  # Individual episode rewards
                self.avg_rewards = []      # Moving average rewards
                self.success_rates = []
                self.exploration_rates = []
                self.timesteps = []
                
                # Smoothing for success rate
                self.smoothed_success_rate = 0.0
                self.alpha = 0.1  # Smoothing factor (0.1 = smooth, 0.9 = responsive)
                
                # UI placeholders (passed from outside)
                self.status_text = status_text
                self.progress_bar = progress_bar
                self.chart_reward = chart_reward
                self.chart_success = chart_success
                self.chart_exploration = chart_exploration
                    
            def _on_step(self) -> bool:
                # Update progress
                percent = min(self.num_timesteps / total_timesteps, 1.0)
                with self.progress_bar:
                    st.progress(percent)
                self.status_text.text(f"正在训练神经网络... 进度: {self.num_timesteps}/{total_timesteps} 步")
                
                # Capture metrics every 200 steps to reduce overhead
                if self.num_timesteps % 200 == 0:
                    # Track both episode reward and average reward
                    if len(self.model.ep_info_buffer) > 0:
                        # Get the most recent episode reward (latest completed episode)
                        latest_ep_reward = self.model.ep_info_buffer[-1]['r']
                        self.episode_rewards.append(latest_ep_reward)
                        
                        # Calculate moving average
                        avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                        self.avg_rewards.append(avg_reward)
                        
                        # Success rate: episodes with reward > 0 (successful landing)
                        # Only calculate if buffer has enough data
                        if len(self.model.ep_info_buffer) >= 5:
                            success_count = sum(1 for ep in self.model.ep_info_buffer if ep['r'] > 0)
                            raw_success_rate = (success_count / len(self.model.ep_info_buffer)) * 100
                            
                            # Apply exponential moving average for smoothing
                            self.smoothed_success_rate = self.alpha * raw_success_rate + (1 - self.alpha) * self.smoothed_success_rate
                            self.success_rates.append(self.smoothed_success_rate)
                        else:
                            # Not enough data yet
                            self.success_rates.append(0)
                    else:
                        self.episode_rewards.append(0)
                        self.avg_rewards.append(0)
                        self.success_rates.append(0)
                    
                    # Exploration rate (linear decay)
                    exploration_rate = self.model.exploration_rate
                    self.exploration_rates.append(exploration_rate)
                    
                    self.timesteps.append(self.num_timesteps)
                    
                    # Update reward chart with both lines
                    if len(self.episode_rewards) > 1:
                        df_reward = pd.DataFrame({
                            "steps": self.timesteps, 
                            "当前奖励": self.episode_rewards,
                            "平均奖励": self.avg_rewards
                        }).set_index("steps")
                        self.chart_reward.line_chart(df_reward, height=200, color=["#1f77b4", "#cccccc"])
                        
                        df_success = pd.DataFrame({"steps": self.timesteps, "success_rate": self.success_rates}).set_index("steps")
                        self.chart_success.line_chart(df_success, height=200)
                        
                        df_exploration = pd.DataFrame({"steps": self.timesteps, "exploration_rate": self.exploration_rates}).set_index("steps")
                        self.chart_exploration.line_chart(df_exploration, height=200)
                        
                return True

        # Init Environment & Model
        env = gym.make("LunarLander-v3", render_mode=None)
        
        # Map network size selection to architecture (research-backed sizes)
        net_arch_map = {
            "简单 (128-128)": [128, 128],
            "标准 (256-256) 推荐": [256, 256],  # Proven successful for LunarLander
            "复杂 (512-256)": [512, 256]
        }
        policy_kwargs = dict(net_arch=net_arch_map[network_size])
        
        model = DQN(
            "MlpPolicy", 
            env, 
            policy_kwargs=policy_kwargs,  # Custom network architecture
            learning_rate=lr,
            gamma=gamma,
            exploration_initial_eps=exploration_initial,
            exploration_final_eps=exploration_final,
            exploration_fraction=exploration_fraction,
            batch_size=batch_size,
            buffer_size=100000,  # Increased from 50000 for better retention
            learning_starts=1000,  # Start learning after collecting some experiences
            train_freq=4,  # Train every 4 steps
            gradient_steps=1,  # One gradient step per training
            target_update_interval=1000, # Update target network every 1000 steps
            verbose=0,
            device="auto"
        )
        
        status_text.write("正在初始化环境和神经网络...")
        callback = StreamlitCallback(status_text, progress_bar, chart_reward, chart_success, chart_exploration)
        
        # Start Training
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        status_text.success(f"✅ 训练完成！(共 {total_timesteps} 步)")
        progress_bar.empty()  # Clear progress bar
        
        # Save training data to session state
        st.session_state.dqn_training_data = {
            "steps": callback.timesteps,
            "episode_reward": callback.episode_rewards,
            "avg_reward": callback.avg_rewards,
            "success_rate": callback.success_rates,
            "exploration_rate": callback.exploration_rates
        }
        
        # Save to session state
        st.session_state.dqn_model = model
        st.session_state.dqn_source = "student"
        
        # Force rerun to show test section
        time.sleep(1)
        st.rerun()

    # --- Test Section (Only visible if model exists) ---
    if st.session_state.get('dqn_model') is not None and st.session_state.get('dqn_source') == 'student':
        st.divider()
        st.markdown("### 🎥 成果验收")
        st.info('训练结束了！让我们看看这个"新手"机器人的表现如何。')
        
        run_test_btn = st.button("▶️ 运行测试 (10轮)", use_container_width=True)
        
        if run_test_btn:
            model = st.session_state.dqn_model
            env = gym.make("LunarLander-v3", render_mode="rgb_array")
            
            col_anim, col_stats = st.columns([2, 1])
            with col_anim:
                frame_placeholder = st.empty()
            with col_stats:
                st.markdown("#### 📊 实时统计")
                metric_success = st.empty()
                metric_reward = st.empty()
                metric_steps = st.empty()
            
            # Run 10 episodes
            success_count = 0
            total_reward = 0
            total_steps = 0
            
            for episode in range(10):
                obs, _ = env.reset()
                done = False
                steps = 0
                ep_reward = 0
                
                while not done and steps < 500:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    steps += 1
                    
                    if steps % 2 == 0:
                        frame = env.render()
                        # Show timeout warning if approaching limit
                        status = f"Episode {episode+1}/10 | Step {steps}"
                        if steps >= 400:
                            status += " ⚠️ 接近超时"
                        frame_placeholder.image(frame, caption=status, use_container_width=True)
                
                # Update stats
                if ep_reward > 200:
                    success_count += 1
                total_reward += ep_reward
                total_steps += steps
                
                # Update metrics
                current_success_rate = (success_count / (episode + 1)) * 100
                current_avg_reward = total_reward / (episode + 1)
                current_avg_steps = total_steps / (episode + 1)
                
                metric_success.metric("成功率", f"{current_success_rate:.0f}% ({success_count}/{episode+1})")
                metric_reward.metric("平均奖励", f"{current_avg_reward:.1f}")
                metric_steps.metric("平均步数", f"{current_avg_steps:.1f}")
            
            env.close()
            
            # Final evaluation
            final_success_rate = (success_count / 10) * 100
            if final_success_rate >= 80:
                st.balloons()
                st.success("🎉 表现优秀！这个模型已经学会了不少技巧。")
            elif final_success_rate >= 50:
                st.info("😐 表现一般。可以尝试调整参数或增加训练步数。")
            else:
                st.error("💥 表现较差。建议增加训练步数或调整学习率。")
