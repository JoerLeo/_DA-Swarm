import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ==========================================
# 配置与环境 (保持一致)
# ==========================================
@dataclass
class SimParams:
    dt: float = 0.1
    steps: int = 700  # 跑久一点，覆盖8字形全过程
    n_robot: int = 20
    r_body: float = 3.0
    W: int = 120
    H: int = 120

    # 基础参数
    base_k_goal: float = 2.0
    base_k_rep: float = 6.0
    base_k_obs: float = 10.0
    base_v_max: float = 3.0

    # 自适应阈值
    dist_panic: float = 8.0
    dist_lag: float = 35.0


def get_obstacles():
    return [
        {'pos': np.array([40.0, 50.0]), 'r': 6.0},
        {'pos': np.array([80.0, 70.0]), 'r': 7.0},
        {'pos': np.array([60.0, 20.0]), 'r': 5.0},
    ]


def get_target_center(t, W, H):
    # 8字形
    cx = W / 2 + 30.0 * np.sin(0.04 * t)
    cy = H / 2 + 15.0 * np.sin(0.08 * t)
    return np.array([cx, cy])


# ==========================================
# 核心仿真引擎 (支持两种模式)
# ==========================================
def run_simulation(mode='fixed'):
    """
    mode: 'fixed' (基线) 或 'adaptive' (自适应)
    返回: 历史数据字典
    """
    params = SimParams()
    rng = np.random.default_rng(42)  # 固定种子，保证初始位置一样，公平对比
    obstacles = get_obstacles()

    pos = np.zeros((params.n_robot, 2))
    pos[:, 0] = rng.uniform(10, params.W - 10, params.n_robot)
    pos[:, 1] = rng.uniform(params.H - 20, params.H - 5, params.n_robot)
    vel = np.zeros_like(pos)

    # 相对目标形状
    angles = np.linspace(0, 2 * np.pi, params.n_robot, endpoint=False)
    rel_goals = np.column_stack([20.0 * np.cos(angles), 20.0 * np.sin(angles)])

    # --- 数据记录 ---
    history = {
        'min_obs_dist': [],  # 安全指标：最近障碍物距离
        'avg_tracking_err': [],  # 任务指标：平均追踪误差
        'avg_k_rep': [],  # 适应性指标：平均斥力系数
        'collisions': 0  # 碰撞计数
    }

    print(f"🏃 Running Simulation: MODE = {mode.upper()} ...")

    for step in range(params.steps):
        center_curr = get_target_center(step, params.W, params.H)
        current_goals = center_curr + rel_goals
        new_vel = np.zeros_like(vel)

        step_k_reps = []  # 记录这一步所有机器人的k_rep
        min_d_obs_step = float('inf')

        tracking_errs = []

        for i in range(params.n_robot):
            p = pos[i]

            # --- 1. 感知与决策 ---
            d_obs_min = float('inf')
            for obs in obstacles:
                d = np.linalg.norm(p - obs['pos']) - obs['r'] - params.r_body
                if d < d_obs_min: d_obs_min = d

            if d_obs_min < min_d_obs_step: min_d_obs_step = d_obs_min
            if d_obs_min < 0: history['collisions'] += 1  # 发生碰撞

            d_target_center = np.linalg.norm(p - center_curr)
            tracking_errs.append(np.linalg.norm(p - current_goals[i]))

            # 默认参数
            k_g_mult = 1.0
            k_r_mult = 1.0
            v_limit = params.base_v_max

            if mode == 'adaptive':
                # Day 2 的逻辑
                if d_obs_min < params.dist_panic:  # 恐慌
                    k_r_mult = 5.0
                    k_g_mult = 0.1
                elif d_target_center > params.dist_lag:  # 追击
                    k_g_mult = 2.5
                    v_limit = params.base_v_max * 1.5

            step_k_reps.append(params.base_k_rep * k_r_mult)

            # --- 2. 力计算 ---
            # 引力
            g_vec = current_goals[i] - p
            d_g = np.linalg.norm(g_vec)
            if d_g > 0:
                v_goal = (g_vec / d_g) * v_limit * 0.8 + g_vec * (params.base_k_goal * k_g_mult) * 0.1
            else:
                v_goal = np.zeros(2)

            # 障碍物斥力
            v_rep_obs = np.zeros(2)
            for obs in obstacles:
                diff = p - obs['pos']
                dist = np.linalg.norm(diff)
                safe_d = obs['r'] + params.r_body + 1.0
                # 自适应模式下，感知范围随恐慌程度略微扩大效果更好，这里简化处理
                sense_range = safe_d + params.dist_panic
                if dist < sense_range:
                    mag = (params.base_k_obs * k_r_mult) * (1.0 / max(dist - obs['r'], 0.1) - 1.0 / sense_range)
                    if mag > 0: v_rep_obs += (diff / dist) * mag

            # 队友斥力
            v_rep_bot = np.zeros(2)
            for j in range(params.n_robot):
                if i == j: continue
                diff = p - pos[j]
                d = np.linalg.norm(diff)
                if d < params.r_body * 2 + 2.0:
                    mag = params.base_k_rep * (1.0 / d - 1.0 / (params.r_body * 2 + 2.0))
                    v_rep_bot += (diff / d) * mag

            total_v = v_goal + v_rep_obs + v_rep_bot
            s = np.linalg.norm(total_v)
            if s > v_limit: total_v = (total_v / s) * v_limit
            new_vel[i] = total_v

        pos += new_vel * params.dt
        pos[:, 0] = np.clip(pos[:, 0], 0, params.W)
        pos[:, 1] = np.clip(pos[:, 1], 0, params.H)
        vel = new_vel

        # 记录每一步的数据
        history['min_obs_dist'].append(min_d_obs_step)
        history['avg_tracking_err'].append(np.mean(tracking_errs))
        history['avg_k_rep'].append(np.mean(step_k_reps))

    return history


# ==========================================
# 绘图分析
# ==========================================
def plot_results(hist_fixed, hist_adaptive):
    t = np.arange(len(hist_fixed['min_obs_dist'])) * 0.1

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 1. 安全性对比 (最近障碍物距离)
    ax1 = axs[0]
    ax1.plot(t, hist_fixed['min_obs_dist'], 'k--', label='Baseline (Fixed)', alpha=0.7)
    ax1.plot(t, hist_adaptive['min_obs_dist'], 'r-', label='Ours (Adaptive)', linewidth=2)
    ax1.axhline(0, color='red', linestyle=':', label='Collision Threshold')
    ax1.set_ylabel('Min Dist to Obstacle')
    ax1.set_title('Metric 1: Safety Analysis (Collision Avoidance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 任务表现对比 (追踪误差)
    ax2 = axs[1]
    ax2.plot(t, hist_fixed['avg_tracking_err'], 'k--', label='Baseline', alpha=0.7)
    ax2.plot(t, hist_adaptive['avg_tracking_err'], 'b-', label='Adaptive', linewidth=2)
    ax2.set_ylabel('Tracking Error (pixels)')
    ax2.set_title('Metric 2: Task Performance (Formation Keeping)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 自适应响应机制 (k_rep 变化)
    ax3 = axs[2]
    # 基线是直的
    ax3.plot(t, hist_fixed['avg_k_rep'], 'k--', label='Fixed $k_{rep}$')
    # 自适应是波动的
    ax3.plot(t, hist_adaptive['avg_k_rep'], 'g-', label='Adaptive $k_{rep}$ Response', linewidth=1.5)
    ax3.fill_between(t, 0, hist_adaptive['avg_k_rep'], color='green', alpha=0.1)
    ax3.set_ylabel('Repulsion Coeff ($k_{rep}$)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Mechanism: Adaptive Parameter Response')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300)
    print(f"\n📊 图表已生成: experiment_comparison.png")

    # 打印最终统计
    print("-" * 40)
    print(f"Final Statistics Comparison:")
    print(f"Baseline Collisions: {hist_fixed['collisions']}")
    print(f"Adaptive Collisions: {hist_adaptive['collisions']} (Should be lower!)")
    print("-" * 40)


if __name__ == "__main__":
    # 跑两遍
    h_fixed = run_simulation(mode='fixed')
    h_adaptive = run_simulation(mode='adaptive')

    # 画图
    plot_results(h_fixed, h_adaptive)