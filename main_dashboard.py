import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec
from dataclasses import dataclass


# ==========================================
# 1. 基础参数与配置
# ==========================================
@dataclass
class DashboardParams:
    dt: float = 0.1
    steps: int = 400  # 仿真步数
    n_robot: int = 20  # 机器人数
    r_body: float = 3.0  # 机器人半径
    W: int = 120  # 画布宽
    H: int = 120  # 画布高

    # 基础控制参数
    base_k_rep: float = 6.0
    base_k_goal: float = 2.0

    # 状态切换阈值
    dist_panic: float = 8.0  # 离障碍物多近开始恐慌
    dist_lag: float = 35.0  # 离目标多远算掉队


def get_obstacles():
    """定义三个障碍物"""
    return [
        {'pos': np.array([40.0, 50.0]), 'r': 6.0},
        {'pos': np.array([80.0, 70.0]), 'r': 7.0},
        {'pos': np.array([60.0, 20.0]), 'r': 5.0},
    ]


def get_target_center(t, W, H):
    """生成8字形动态轨迹中心"""
    return np.array([W / 2 + 25 * np.sin(0.05 * t), H / 2 + 15 * np.cos(0.03 * t)])


# ==========================================
# 2. 核心逻辑：运行仿真与生成仪表盘
# ==========================================
def run_dashboard_demo():
    # --- A. 初始化 ---
    params = DashboardParams()
    rng = np.random.default_rng(42)
    obstacles = get_obstacles()

    # 随机初始位置 (在画布中间随机撒点)
    pos = np.random.rand(params.n_robot, 2) * 60 + 30
    vel = np.zeros_like(pos)

    # 预计算相对目标形状 (圆形)
    angles = np.linspace(0, 2 * np.pi, params.n_robot, endpoint=False)
    rel_goals = np.column_stack([20.0 * np.cos(angles), 20.0 * np.sin(angles)])

    # 历史数据容器
    history = {
        'pos': [],
        'center': [],
        'm1': [], 'm2': [], 'm3': [], 'm4': [], 'score': [],
        'colors': []  # <--- 关键：用来存每一帧的颜色
    }

    print("🚀 正在计算仿真数据 (Pre-computing)...")

    # --- B. 仿真循环 ---
    for step in range(params.steps):
        center_curr = get_target_center(step, params.W, params.H)
        current_goals = center_curr + rel_goals

        new_vel = np.zeros_like(vel)
        step_colors = []  # <--- 1. 每一步创建一个空列表存颜色

        for i in range(params.n_robot):
            p = pos[i]

            # 1. 感知环境
            min_obs_d = float('inf')
            for obs in obstacles:
                d = np.linalg.norm(p - obs['pos']) - obs['r']
                min_obs_d = min(min_obs_d, d)

            dist_to_target = np.linalg.norm(p - center_curr)

            # 2. 决策与变色 (核心逻辑)
            # 默认：巡航模式 (绿色)
            kr, kg = params.base_k_rep, params.base_k_goal
            color = '#32CD32'  # LimeGreen

            if min_obs_d < params.dist_panic:
                # 避险模式 (红色)
                kr *= 5.0
                kg *= 0.1
                color = '#FF0000'  # Red
            elif dist_to_target > params.dist_lag:
                # 追击模式 (金色)
                kg *= 2.5
                color = '#FFD700'  # Gold

            step_colors.append(color)  # <--- 2. 把颜色存进去

            # 3. 力计算
            v_sum = np.zeros(2)
            # 引力
            g_vec = current_goals[i] - p
            v_sum += g_vec * kg * 0.1

            # 障碍物斥力
            for obs in obstacles:
                diff = p - obs['pos']
                d_o = np.linalg.norm(diff)
                safe_dist = obs['r'] + params.dist_panic + 3.0
                if d_o < safe_dist:
                    # 斥力公式
                    v_sum += (diff / d_o) * kr * (1.0 / (d_o - obs['r']) - 1.0 / safe_dist) * 5.0

            # 队友斥力
            for j in range(params.n_robot):
                if i != j:
                    diff = p - pos[j]
                    d_b = np.linalg.norm(diff)
                    if d_b < 8.0:
                        v_sum += (diff / d_b) * 6.0 * (1.0 / d_b - 1.0 / 8.0)

            # 限速
            s = np.linalg.norm(v_sum)
            limit = 3.0 if color != '#FFD700' else 4.5  # 追击时允许超速
            if s > limit: v_sum = v_sum / s * limit
            new_vel[i] = v_sum

        # 更新位置
        pos += new_vel * params.dt

        # 4. [防穿模] 刚体强制约束 (防止卡在球里)
        for i in range(params.n_robot):
            for obs in obstacles:
                diff = pos[i] - obs['pos']
                dist = np.linalg.norm(diff)
                min_allowed = obs['r'] + params.r_body + 0.1
                if dist < min_allowed:
                    if dist > 0:
                        normal = diff / dist
                    else:
                        normal = np.array([1.0, 0.0])
                    # 强制推到表面
                    pos[i] = obs['pos'] + normal * min_allowed
                    new_vel[i] *= 0.5  # 撞墙减速

        # 边界限制
        pos[:, 0] = np.clip(pos[:, 0], 0, params.W)
        pos[:, 1] = np.clip(pos[:, 1], 0, params.H)
        vel = new_vel

        # 5. 指标计算 (优化版，不那么严苛)
        target_radius = 20.0

        # M2: 安全性 (按比例给分)
        safe_count = 0
        for i in range(params.n_robot):
            is_safe = True
            for obs in obstacles:
                if np.linalg.norm(pos[i] - obs['pos']) < obs['r'] + params.r_body - 0.5:
                    is_safe = False;
                    break
            if is_safe: safe_count += 1
        m2_safety = safe_count / params.n_robot

        # M1: 覆盖率 (放宽半径判定到 +/- 12)
        dists = np.linalg.norm(pos - center_curr, axis=1)
        m1 = np.sum((dists > target_radius - 12.0) & (dists < target_radius + 12.0)) / params.n_robot

        compactness = np.sum(dists < target_radius + 15.0) / params.n_robot
        m2 = 0.7 * m2_safety + 0.3 * compactness

        # M3: 均匀性 (放宽标准差惩罚)
        if params.n_robot > 1:
            nn_dists = [np.min([np.linalg.norm(pos[k] - pos[j]) for j in range(params.n_robot) if k != j]) for k in
                        range(params.n_robot)]
            m3 = np.exp(-np.std(nn_dists) / 12.0)
        else:
            m3 = 0.0

        # M4: 极化度
        speeds = np.linalg.norm(vel, axis=1) + 1e-6
        mean_speed = np.mean(speeds)
        m4 = np.linalg.norm(np.sum(vel, axis=0)) / (params.n_robot * mean_speed) if mean_speed > 0.1 else 0.0

        # 最终得分
        final_score = 0.35 * m2 + 0.30 * m3 + 0.20 * m1 + 0.15 * m4

        # 6. 保存所有数据到 History
        history['pos'].append(pos.copy())
        history['center'].append(center_curr)
        history['colors'].append(step_colors)  # <--- 保存本帧颜色列表
        history['m1'].append(m1);
        history['m2'].append(m2)
        history['m3'].append(m3);
        history['m4'].append(m4)
        history['score'].append(final_score)

    # === C. 绘制仪表盘动画 ===
    print("🎥 正在渲染终极版仪表盘 (Colors + Score)...")

    fig = plt.figure(figsize=(14, 10))
    # 5行3列布局
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.6, wspace=0.25)

    # 左侧大图
    ax_sim = fig.add_subplot(gs[:, :2])
    ax_sim.set_xlim(0, params.W);
    ax_sim.set_ylim(params.H, 0)
    ax_sim.set_title("Simulation View: Dynamic Adaptive Swarm", fontsize=14, fontweight='bold')

    # 左上角分数显示
    score_text = ax_sim.text(0.02, 0.98, '', transform=ax_sim.transAxes,
                             fontsize=12, fontweight='bold', color='purple',
                             verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="purple", alpha=0.8))

    # 右侧5个小图
    ax_m1 = fig.add_subplot(gs[0, 2])
    ax_m2 = fig.add_subplot(gs[1, 2])
    ax_m3 = fig.add_subplot(gs[2, 2])
    ax_m4 = fig.add_subplot(gs[3, 2])
    ax_score = fig.add_subplot(gs[4, 2])

    metrics_axes = [ax_m1, ax_m2, ax_m3, ax_m4, ax_score]
    titles = ["M1: Coverage (0.20)", "M2: Safety (0.35)", "M3: Uniformity (0.30)", "M4: Polarization (0.15)",
              "★ Final Weighted Score"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    metric_keys = ['m1', 'm2', 'm3', 'm4', 'score']
    lines = []

    # 初始化右侧曲线
    for i, ax in enumerate(metrics_axes):
        ax.set_title(titles[i], fontsize=10, pad=3)
        lw = 2.5 if i == 4 else 1.5
        line, = ax.plot([], [], color=colors[i], lw=lw)
        lines.append(line)
        ax.set_xlim(0, params.steps);
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3);
        ax.tick_params(axis='y', labelsize=8)
        # 只在最后一张图显示X轴刻度
        if i < 4:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=8); ax.set_xlabel("Time Steps", fontsize=9); ax.set_facecolor('#f9f2ff')

    # 初始化散点图 (初始给个颜色避免未定义)
    sc = ax_sim.scatter([], [], color='green', s=60, edgecolors='k', zorder=5)

    # 画目标圆环和障碍物
    target_circle = plt.Circle((0, 0), 20, color='blue', fill=False, lw=2, alpha=0.6)
    ax_sim.add_patch(target_circle)

    for obs in obstacles:
        ax_sim.add_patch(plt.Circle(obs['pos'], obs['r'], color='black', alpha=0.7))
        ax_sim.add_patch(
            plt.Circle(obs['pos'], obs['r'] + params.dist_panic, color='red', fill=False, ls='--', alpha=0.3))

    def init():
        sc.set_offsets(np.empty((0, 2)))
        score_text.set_text("")
        for line in lines: line.set_data([], [])
        return [sc, target_circle, score_text] + lines

    def update(frame):
        # 1. 更新机器人位置
        sc.set_offsets(history['pos'][frame])

        # 2. [关键修复] 更新颜色 (Face Colors)
        # 从历史记录取出当前帧的颜色列表
        current_colors = history['colors'][frame]
        sc.set_facecolors(current_colors)
        sc.set_edgecolors('k')  # 强制黑边，防止边框消失

        # 3. 更新目标位置
        target_circle.set_center(history['center'][frame])

        # 4. 更新分数文字
        score_text.set_text(f"Current Score: {history['score'][frame]:.3f}")

        # 5. 更新右侧曲线
        x_data = np.arange(frame + 1)
        for i, key in enumerate(metric_keys):
            lines[i].set_data(x_data, history[key][:frame + 1])

        return [sc, target_circle, score_text] + lines

    ani = FuncAnimation(fig, update, frames=len(history['pos']), init_func=init, interval=30, blit=True)

    # 保存 GIF
    ani.save("dashboard_final_complete.gif", writer=PillowWriter(fps=30))
    plt.close(fig)
    print("✅ 全部完成！文件已保存: dashboard_final_complete.gif")


if __name__ == "__main__":
    run_dashboard_demo()