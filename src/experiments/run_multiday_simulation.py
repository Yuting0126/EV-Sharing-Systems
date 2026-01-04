"""
multi_day_simulation.py
=======================
多日滚动仿真：证明车辆调度的必要性

对比两种策略在 7 天内的表现：
1. With Relocation: 每天结束后调度，第二天从平衡状态开始
2. No Relocation: 不调度，不平衡累积到第二天

Author: [Your Name]
Date: 2024-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import copy

from main_experiment_gurobi import (
    SystemConfig, GridData, TrafficData,
    load_real_data, solve_miqp_gurobi, solve_relocation_subproblem,
    GUROBI_AVAILABLE
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass


# ============================================================
# MODIFIED MIQP: 可指定初始状态
# ============================================================

def solve_miqp_with_initial_state(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData,
    N_initial: np.ndarray,  # shape: (NS,) 初始车辆数
    E_initial: np.ndarray,  # shape: (NS,) 初始能量
    enforce_periodic: bool = True  # 是否强制周期性约束
) -> Dict:
    """
    求解 MIQP，可指定任意初始状态。
    """
    if not GUROBI_AVAILABLE:
        return None
    
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    E_min = config.E_min
    b = config.b_elasticity
    N_max = config.N_max
    e_max = config.e_max_per_period
    
    a = traffic_data.a_demand
    L = traffic_data.L_energy
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    model = gp.Model("EV_Sharing_MIQP")
    model.Params.OutputFlag = 0  # 静默模式
    model.Params.TimeLimit = 60
    
    # 决策变量
    f = model.addVars(NS, NS, T, lb=0, name="f")
    N = model.addVars(NS, T + 1, lb=0, ub=N_max, name="N")
    E = model.addVars(NS, T + 1, lb=0, name="E")
    e = model.addVars(NS, T, lb=0, name="e")
    
    # 目标函数
    revenue = gp.quicksum(
        (a[i, j, t] * f[i, j, t] - f[i, j, t] * f[i, j, t]) / b
        for i in range(NS) for j in range(NS) for t in range(T)
        if a[i, j, t] > 0
    )
    charging_cost = gp.quicksum(
        lmp[mapping.get(i, 0), t] * e[i, t]
        for i in range(NS) for t in range(T)
    )
    model.setObjective(revenue - charging_cost, GRB.MAXIMIZE)
    
    # 约束
    # 1. 需求上界
    for i in range(NS):
        for j in range(NS):
            for t in range(T):
                if a[i, j, t] > 0:
                    model.addConstr(f[i, j, t] <= a[i, j, t])
                else:
                    model.addConstr(f[i, j, t] == 0)
    
    # 2. 初始条件 (使用传入的状态)
    for i in range(NS):
        model.addConstr(N[i, 0] == N_initial[i])
        model.addConstr(E[i, 0] == E_initial[i])
    
    # 3. 车辆守恒
    for i in range(NS):
        for t in range(T):
            inflow = gp.quicksum(f[j, i, t] for j in range(NS))
            outflow = gp.quicksum(f[i, j, t] for j in range(NS))
            model.addConstr(N[i, t + 1] == N[i, t] + inflow - outflow)
    
    # 4. 能量守恒
    for i in range(NS):
        for t in range(T):
            energy_out = gp.quicksum(E_max * f[i, j, t] for j in range(NS))
            energy_in = gp.quicksum((E_max - L[j, i, t]) * f[j, i, t] for j in range(NS))
            model.addConstr(E[i, t + 1] == E[i, t] + e[i, t] - energy_out + energy_in)
    
    # 5. 能量上下界
    for i in range(NS):
        for t in range(T + 1):
            model.addConstr(E[i, t] <= E_max * N[i, t])
    
    # 6. 充电功率限制
    for i in range(NS):
        for t in range(T):
            model.addConstr(e[i, t] <= e_max * N[i, t])
    
    # 7. 周期性条件 (可选)
    if enforce_periodic:
        for i in range(NS):
            model.addConstr(N[i, T] == N_initial[i])
    
    # 求解
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # 提取结果
        f_result = np.zeros((NS, NS, T))
        N_result = np.zeros((NS, T + 1))
        E_result = np.zeros((NS, T + 1))
        e_result = np.zeros((NS, T))
        
        for i in range(NS):
            for t in range(T + 1):
                N_result[i, t] = N[i, t].X
                E_result[i, t] = E[i, t].X
            for t in range(T):
                e_result[i, t] = e[i, t].X
                for j in range(NS):
                    f_result[i, j, t] = f[i, j, t].X
        
        # 计算收益
        c_result = np.zeros((NS, NS, T))
        total_revenue = 0
        for i in range(NS):
            for j in range(NS):
                for t in range(T):
                    if a[i, j, t] > 0:
                        c_result[i, j, t] = (a[i, j, t] - f_result[i, j, t]) / b
                        total_revenue += c_result[i, j, t] * f_result[i, j, t]
        
        total_charging_cost = 0
        for i in range(NS):
            for t in range(T):
                bus_id = mapping.get(i, 0)
                total_charging_cost += lmp[bus_id, t] * e_result[i, t]
        
        return {
            'flow': f_result,
            'vehicles': N_result,
            'energy': E_result,
            'charging': e_result,
            'revenue': total_revenue,
            'charging_cost': total_charging_cost,
            'profit': model.objVal,
            'final_N': N_result[:, -1],  # 终态车辆
            'final_E': E_result[:, -1],  # 终态能量
        }
    else:
        return None


# ============================================================
# 启发式日仿真 (用于无调度场景，快速)
# ============================================================

def simulate_day_heuristic(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData,
    N_initial: np.ndarray,
    E_initial: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    用启发式方法仿真一天的运营（不调度）。
    使用固定定价策略，快速计算利润和车辆变化。
    
    Returns:
        daily_profit, final_N, final_E, imbalance
    """
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    b = config.b_elasticity
    
    a = traffic_data.a_demand
    L = traffic_data.L_energy
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    # 使用平均价格
    avg_price = 10.0
    
    N_current = N_initial.copy()
    E_current = E_initial.copy()
    
    total_revenue = 0
    total_charging_cost = 0
    
    for t in range(T):
        # 计算需求
        f_flow = np.maximum(0, a[:, :, t] - b * avg_price)
        
        # 检查车辆可用性并限制流量
        for i in range(NS):
            outflow = np.sum(f_flow[i, :])
            if outflow > N_current[i]:
                scale = N_current[i] / outflow if outflow > 0 else 0
                f_flow[i, :] *= scale
        
        # 更新车辆数
        for i in range(NS):
            inflow = np.sum(f_flow[:, i])
            outflow = np.sum(f_flow[i, :])
            N_current[i] = max(0, N_current[i] + inflow - outflow)
        
        # 更新能量和充电
        for i in range(NS):
            energy_out = E_max * np.sum(f_flow[i, :])
            energy_in = np.sum((E_max - L[:, i, t]) * f_flow[:, i])
            E_current[i] = E_current[i] - energy_out + energy_in
            
            # 需要充电
            target = E_max * N_current[i]
            charge = max(0, target - E_current[i])
            E_current[i] += charge
            
            bus_id = mapping.get(i, 0)
            total_charging_cost += lmp[bus_id, t] * charge
        
        # 收益
        for i in range(NS):
            for j in range(NS):
                total_revenue += avg_price * f_flow[i, j]
    
    daily_profit = total_revenue - total_charging_cost
    imbalance = np.std(N_current)
    
    return daily_profit, N_current, E_current, imbalance


# ============================================================
# 多日滚动仿真
# ============================================================

def run_multi_day_simulation(num_days: int = 7):
    """
    运行多日滚动仿真，对比有调度和无调度的累积效果。
    使用真实的每日不同需求数据。
    """
    print("\n" + "=" * 70)
    print(f"MULTI-DAY SIMULATION: {num_days} Days (with REAL daily demand variation)")
    print("=" * 70 + "\n")
    
    # 加载基础配置
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    NS = config.num_stations
    
    # 加载 7 天的 OD 数据
    import os
    od_week_file = "data/processed/od_matrix_week.npy"
    if os.path.exists(od_week_file):
        od_week = np.load(od_week_file)  # shape: (10, 10, 24, 7)
        print(f"Loaded weekly OD data: shape {od_week.shape}")
    else:
        print("Warning: Weekly OD data not found. Using repeated single-day data.")
        od_week = np.stack([traffic_data.a_demand] * 7, axis=3)
    
    # 初始状态 (增加初始车辆以匹配需求规模)
    N_init = np.full(NS, 800)  # 每站800辆，总8000辆
    E_init = N_init * config.E_max * 0.8
    
    # 记录每日结果
    results_with_reloc = []
    results_no_reloc = []
    
    # 当前状态 (分别追踪)
    N_current_with = N_init.copy()
    E_current_with = E_init.copy()
    N_current_no = N_init.copy()
    E_current_no = E_init.copy()
    
    print("Running heuristic simulations with daily-varying demand...\n")
    
    for day in range(1, num_days + 1):
        day_idx = day - 1  # 0-indexed
        
        # 获取当天的需求数据
        daily_demand = od_week[:, :, :, day_idx].astype(float)  # (NS, NS, 24)
        daily_traffic = TrafficData(
            a_demand=daily_demand,
            L_energy=traffic_data.L_energy,
            station_to_bus=traffic_data.station_to_bus
        )
        
        day_trips = daily_demand.sum()
        print(f"--- Day {day} ({['Tue','Wed','Thu','Fri','Sat','Sun','Mon'][day_idx]}) | Trips: {day_trips:,.0f} ---")
        
        # === 有调度策略 ===
        # 使用当前状态运行一天，然后晚上调度恢复到平衡
        daily_profit_with, N_end_with, E_end_with, imb_with = simulate_day_heuristic(
            config, grid_data, daily_traffic,
            N_current_with, E_current_with
        )
        
        # 计算夜间调度成本并实际执行调度 (恢复到平衡状态)
        target = np.mean(N_end_with)  # 目标：平均分布
        N_balanced = np.full(NS, target)
        
        # 只计算需要移动的车辆 (单边计算，因为一辆车只需移动一次)
        vehicles_to_move = np.sum(np.maximum(N_end_with - target, 0))  # 超额站点移出
        reloc_cost = vehicles_to_move * 5.0  # $5/辆
        daily_profit_with -= reloc_cost
        
        # **关键：调度后真正恢复到平衡状态**
        N_current_with = N_balanced.copy()  # 实际调度到平衡
        E_current_with = E_end_with.copy()  # 电量保持不变
        
        results_with_reloc.append({
            'day': day,
            'profit': daily_profit_with,
            'relocation_cost': reloc_cost,
            'imbalance': 0.0  # 调度后完全平衡
        })
        print(f"  [With Reloc] Profit: ${daily_profit_with:,.0f}, Reloc Cost: ${reloc_cost:,.0f}")
        
        # === 无调度策略 ===
        # 使用当前状态运行一天，然后不调度，不平衡累积
        daily_profit_no, N_end_no, E_end_no, imb_no = simulate_day_heuristic(
            config, grid_data, daily_traffic,
            N_current_no, E_current_no
        )
        
        # 不调度: 终态直接作为下一天初态
        N_current_no = N_end_no.copy()
        E_current_no = E_end_no.copy()
        
        results_no_reloc.append({
            'day': day,
            'profit': daily_profit_no,
            'relocation_cost': 0,
            'imbalance': imb_no
        })
        print(f"  [No Reloc]   Profit: ${daily_profit_no:,.0f}, Imbalance: {imb_no:.1f}")
    
    # 汇总结果
    df_with = pd.DataFrame(results_with_reloc)
    df_no = pd.DataFrame(results_no_reloc)
    
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    total_profit_with = df_with['profit'].sum()
    total_profit_no = df_no['profit'].sum()
    
    print(f"\nWith Relocation (Total {num_days} days):")
    print(f"  Total Profit: ${total_profit_with:,.0f}")
    print(f"  Avg Daily Profit: ${total_profit_with/num_days:,.0f}")
    print(f"  Total Relocation Cost: ${df_with['relocation_cost'].sum():,.0f}")
    
    print(f"\nNo Relocation (Total {num_days} days):")
    print(f"  Total Profit: ${total_profit_no:,.0f}")
    print(f"  Avg Daily Profit: ${total_profit_no/num_days:,.0f}")
    print(f"  Final Imbalance (std): {df_no['imbalance'].iloc[-1]:.1f}")
    
    print(f"\n** Relocation Advantage: ${total_profit_with - total_profit_no:,.0f} **")
    
    # 保存结果
    df_with.to_csv('multi_day_with_reloc.csv', index=False)
    df_no.to_csv('multi_day_no_reloc.csv', index=False)
    
    # 可视化
    visualize_multi_day(df_with, df_no, num_days)
    
    return df_with, df_no


def visualize_multi_day(df_with: pd.DataFrame, df_no: pd.DataFrame, num_days: int):
    """
    可视化多日仿真结果
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    days = list(range(1, num_days + 1))
    
    # Plot 1: 每日利润对比
    ax1 = axes[0]
    ax1.plot(days, df_with['profit'], 'g-o', label='With Relocation', linewidth=2)
    ax1.plot(days, df_no['profit'], 'r--s', label='No Relocation', linewidth=2)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Daily Profit ($)')
    ax1.set_title('Daily Profit Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 累积利润
    ax2 = axes[1]
    ax2.plot(days, df_with['profit'].cumsum(), 'g-o', label='With Relocation', linewidth=2)
    ax2.plot(days, df_no['profit'].cumsum(), 'r--s', label='No Relocation', linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Cumulative Profit ($)')
    ax2.set_title('Cumulative Profit Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 车辆不平衡度
    ax3 = axes[2]
    ax3.plot(days, df_with['imbalance'], 'g-o', label='With Relocation', linewidth=2)
    ax3.plot(days, df_no['imbalance'], 'r--s', label='No Relocation', linewidth=2)
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Vehicle Imbalance (std)')
    ax3.set_title('Vehicle Distribution Imbalance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_day_comparison.png', dpi=150)
    print("\nChart saved to multi_day_comparison.png")
    # plt.show()  # 注释掉避免阻塞


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    df_with, df_no = run_multi_day_simulation(num_days=7)
