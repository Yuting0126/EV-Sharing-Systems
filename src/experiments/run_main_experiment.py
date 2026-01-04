"""
main_experiment_gurobi.py
==========================
EV Sharing System Optimization - Complete Experimental Script with Gurobi

This script integrates:
1. Data loading (IEEE 33 Grid + Synthetic Traffic Demand)
2. Algorithm 1: Integrated MIQP (Pricing + Charging) using Gurobi
3. Algorithm 2: Benders Decomposition for Relocation (Heuristic)
4. Result Analysis and Visualization

Dependencies:
    pip install numpy pandas matplotlib gurobipy

Author: [Your Name]
Date: 2024-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

# Gurobi imports
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not found. Will use heuristic fallback.")

# ============================================================
# 1. CONFIGURATION & DATA STRUCTURES
# ============================================================

@dataclass
class SystemConfig:
    """系统配置参数"""
    # Time settings
    T: int = 24                  # 时段数 (24小时)
    delta_t: float = 1.0         # 时间步长 (小时)
    
    # EV parameters
    E_max: float = 27.0          # 电池容量 kWh
    E_min: float = 3.0           # 最低电量 kWh
    p_charge: float = 7.0        # 充电功率 kW
    e_max_per_period: float = 7.0  # 单时段最大充电量 kWh
    
    # Demand elasticity: f = a - b * price (降低弹性使价格影响更温和)
    b_elasticity: float = 5.0
    
    # Network sizes
    num_stations: int = 8        # 充电站/交通节点数
    num_buses: int = 33          # 电网节点数
    
    # Initial conditions
    N_init: int = 100            # 每站点初始车辆数
    E_init: float = 2100.0       # 每站点初始总能量 kWh
    N_max: int = 200             # 每站点最大车辆数


@dataclass
class GridData:
    """电网数据"""
    bus_df: pd.DataFrame
    branch_df: pd.DataFrame
    base_load: np.ndarray        # (num_buses, T)
    lmp: np.ndarray              # (num_buses, T) - 简化版使用固定电价


@dataclass
class TrafficData:
    """交通/需求数据"""
    a_demand: np.ndarray         # (NS, NS, T)
    L_energy: np.ndarray         # (NS, NS, T)
    station_to_bus: Dict[int, int]


# ============================================================
# 2. DATA GENERATION
# ============================================================

def generate_synthetic_data(config: SystemConfig) -> Tuple[GridData, TrafficData]:
    """生成仿真数据"""
    NS = config.num_stations
    T = config.T
    NB = config.num_buses
    
    # --- Grid Data ---
    bus_data = {
        'bus_id': list(range(1, NB + 1)),
        'Pd_kW': [0] + [100] * (NB - 1),
        'Qd_kVar': [0] + [50] * (NB - 1)
    }
    bus_df = pd.DataFrame(bus_data)
    branch_df = pd.DataFrame()
    
    # 负荷曲线
    load_profile = np.array([
        0.6, 0.5, 0.45, 0.42, 0.45, 0.55,
        0.7, 0.85, 0.95, 1.0, 0.98, 0.95,
        0.9, 0.88, 0.9, 0.95, 1.0, 0.98,
        0.95, 0.9, 0.85, 0.75, 0.7, 0.65
    ])
    base_load = np.outer(bus_df['Pd_kW'].values, load_profile)
    
    # 简化 LMP (实际应通过 OPF 计算)
    base_price = 0.20
    lmp = base_price * (0.8 + 0.4 * load_profile)
    lmp = np.tile(lmp, (NB, 1))
    
    grid_data = GridData(bus_df=bus_df, branch_df=branch_df, base_load=base_load, lmp=lmp)
    
    # --- Traffic Data ---
    station_to_bus = {0: 4, 1: 7, 2: 11, 3: 14, 4: 18, 5: 22, 6: 26, 7: 30}
    
    # 需求截距 (潮汐模式)
    a_demand = np.zeros((NS, NS, T))
    for t in range(T):
        if 7 <= t <= 9:  # 早高峰
            for i in range(4):
                for j in range(4, 8):
                    a_demand[i, j, t] = 50 + np.random.rand() * 20
        elif 17 <= t <= 19:  # 晚高峰
            for i in range(4, 8):
                for j in range(4):
                    a_demand[i, j, t] = 50 + np.random.rand() * 20
        else:
            a_demand[:, :, t] = 10 + np.random.rand(NS, NS) * 10
        np.fill_diagonal(a_demand[:, :, t], 0)
    
    # 行程能耗
    L_energy = np.zeros((NS, NS, T))
    for i in range(NS):
        for j in range(NS):
            if i != j:
                L_energy[i, j, :] = 5 + abs(i - j) * 2
    
    traffic_data = TrafficData(a_demand=a_demand, L_energy=L_energy, station_to_bus=station_to_bus)
    
    return grid_data, traffic_data


def load_real_data(config: SystemConfig = None) -> Tuple[GridData, TrafficData]:
    """
    加载真实数据 (IEEE 33 + NYC Taxi OD 矩阵)。
    
    需要先运行 preprocess_nyc_taxi.py 生成 OD 矩阵文件。
    
    Returns:
        grid_data: 电网数据
        traffic_data: 交通数据
    """
    import os
    
    print("Loading real data...")
    
    # --- 加载 IEEE 33 电网数据 ---
    bus_file = "data/ieee33_bus.csv"
    branch_file = "data/ieee33_branch.csv"
    
    if not os.path.exists(bus_file):
        raise FileNotFoundError(f"IEEE 33 data not found. Run: python generate_ieee33.py")
    
    bus_df = pd.read_csv(bus_file)
    branch_df = pd.read_csv(branch_file)
    NB = len(bus_df)
    
    # 负荷曲线 (典型日形态)
    load_profile = np.array([
        0.6, 0.5, 0.45, 0.42, 0.45, 0.55,
        0.7, 0.85, 0.95, 1.0, 0.98, 0.95,
        0.9, 0.88, 0.9, 0.95, 1.0, 0.98,
        0.95, 0.9, 0.85, 0.75, 0.7, 0.65
    ])
    base_load = np.outer(bus_df['Pd_kW'].values, load_profile)
    
    # 简化 LMP (与负荷正相关)
    base_price = 0.20
    lmp = base_price * (0.8 + 0.4 * load_profile)
    lmp = np.tile(lmp, (NB, 1))
    
    grid_data = GridData(bus_df=bus_df, branch_df=branch_df, base_load=base_load, lmp=lmp)
    print(f"  Loaded IEEE 33: {NB} buses")
    
    # --- 加载 NYC Taxi OD 矩阵 (使用周数据以保持一致性) ---
    od_file = "data/processed/od_matrix_week.npy"
    
    if not os.path.exists(od_file):
        # Fallback to single day data
        od_file = "data/processed/od_matrix_20190115.npy"
    
    if not os.path.exists(od_file):
        raise FileNotFoundError(f"OD matrix not found. Run: python preprocess_nyc_taxi.py")
    
    od_data = np.load(od_file)
    
    # Handle both formats: (NS, NS, T) or (NS, NS, T, days)
    if len(od_data.shape) == 4:
        # Weekly data: use first day
        od_matrix = od_data[:, :, :, 0]  # shape: (NS, NS, T)
        NS, _, T, _ = od_data.shape
    else:
        od_matrix = od_data
        NS, _, T = od_matrix.shape
    
    print(f"  Loaded OD matrix: {NS} zones, {T} hours, {od_matrix.sum():.0f} total trips (day 1)")
    
    # 将 OD 计数转换为需求截距 a_demand
    # 假设: a = 实际需求 * 弹性系数 (这样在均衡价格下，需求 = 原始计数)
    # 简化: 直接使用 OD 计数作为 a_demand
    a_demand = od_matrix.astype(float)
    
    # 行程能耗: 简化为站点间距离估算
    # 实际应从 Google Maps API 或历史数据获取
    L_energy = np.zeros((NS, NS, T))
    for i in range(NS):
        for j in range(NS):
            if i != j:
                # Assuming 0.2-0.3 kWh/mile. NYC zones are 1-5 miles apart.
                # L = 1-5 kWh is more realistic.
                L_energy[i, j, :] = 1.0 + abs(i - j) * 0.5
    
    # 站点到电网节点的映射
    # IEEE 33 有 33 个节点 (0-32 索引)，我们有 10 个站点，选择负荷较大的末端节点
    station_to_bus = {
        0: 3, 1: 7, 2: 11, 3: 14, 4: 17,
        5: 21, 6: 24, 7: 28, 8: 30, 9: 32
    }
    # 只取前 NS 个映射
    station_to_bus = {i: station_to_bus.get(i, 4) for i in range(NS)}
    
    traffic_data = TrafficData(a_demand=a_demand, L_energy=L_energy, station_to_bus=station_to_bus)
    
    # 更新配置
    if config is not None:
        config.num_stations = NS
        config.num_buses = NB
        # 根据 OD 矩阵规模调整初始车辆数
        avg_hourly_trips = od_matrix.sum() / T
        config.N_init = int(avg_hourly_trips / NS * 0.5)  # 减少车辆数，提高利用率
        config.E_init = config.N_init * config.E_max * 0.8  # 初始 80% 满电
        config.N_max = config.N_init * 3
        print(f"  Updated config: N_init={config.N_init}, E_init={config.E_init:.0f}")
    
    print("  Done loading real data.\n")
    
    return grid_data, traffic_data


# ============================================================
# 3. ALGORITHM 1: INTEGRATED MIQP WITH GUROBI
# ============================================================

def solve_miqp_gurobi(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData
) -> Dict:
    """
    使用 Gurobi 求解集成 MIQP 模型 (定价 + 充电)。
    
    决策变量:
        - c[i,j,t]: 定价 (间接通过 f 计算)
        - f[i,j,t]: 需求流量
        - N[i,t]: 站点车辆数
        - E[i,t]: 站点总能量
        - e[i,t]: 充电量
    
    目标函数:
        max Σ (a/b - f/b) * f - Σ μ * e
        (这是把 c = (a-f)/b 代入后的二次形式)
    
    约束:
        - 需求约束: f ≤ a (非负需求)
        - 车辆守恒
        - 能量守恒
        - 能量上下界
        - 充电功率限制
        - 周期性条件
    """
    print("=" * 60)
    print("ALGORITHM 1: Solving Integrated MIQP with Gurobi")
    print("=" * 60)
    
    if not GUROBI_AVAILABLE:
        print("Gurobi not available. Falling back to heuristic.")
        return solve_miqp_heuristic(config, grid_data, traffic_data)
    
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
    
    # --- 创建 Gurobi 模型 ---
    model = gp.Model("EV_Sharing_MIQP")
    model.Params.OutputFlag = 1  # 显示求解过程
    model.Params.TimeLimit = 300  # 5分钟时限
    
    # --- 决策变量 ---
    # f[i,j,t]: 需求流量
    f = model.addVars(NS, NS, T, lb=0, name="f")
    
    # N[i,t]: 站点车辆数 (t=0 是初始状态)
    N = model.addVars(NS, T + 1, lb=0, ub=N_max, name="N")
    
    # E[i,t]: 站点总能量
    E = model.addVars(NS, T + 1, lb=0, name="E")
    
    # e[i,t]: 充电量
    e = model.addVars(NS, T, lb=0, name="e")
    
    # --- 目标函数 ---
    # 收益: Σ c * f = Σ (a - f) / b * f = Σ (a*f - f^2) / b
    # 充电成本: Σ μ * e
    
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
    
    # --- 约束条件 ---
    
    # 1. 需求上界约束: f <= a
    for i in range(NS):
        for j in range(NS):
            for t in range(T):
                if a[i, j, t] > 0:
                    model.addConstr(f[i, j, t] <= a[i, j, t], name=f"demand_ub_{i}_{j}_{t}")
                else:
                    model.addConstr(f[i, j, t] == 0, name=f"demand_zero_{i}_{j}_{t}")
    
    # 2. 初始条件
    for i in range(NS):
        model.addConstr(N[i, 0] == config.N_init, name=f"N_init_{i}")
        model.addConstr(E[i, 0] == config.E_init, name=f"E_init_{i}")
    
    # 3. 车辆守恒: N[i,t+1] = N[i,t] + 流入 - 流出
    for i in range(NS):
        for t in range(T):
            inflow = gp.quicksum(f[j, i, t] for j in range(NS))
            outflow = gp.quicksum(f[i, j, t] for j in range(NS))
            model.addConstr(
                N[i, t + 1] == N[i, t] + inflow - outflow,
                name=f"vehicle_balance_{i}_{t}"
            )
    
    # 4. 能量守恒: E[i,t+1] = E[i,t] + 充电 - 出发能量 + 到达能量
    for i in range(NS):
        for t in range(T):
            energy_out = gp.quicksum(E_max * f[i, j, t] for j in range(NS))
            energy_in = gp.quicksum((E_max - L[j, i, t]) * f[j, i, t] for j in range(NS))
            model.addConstr(
                E[i, t + 1] == E[i, t] + e[i, t] - energy_out + energy_in,
                name=f"energy_balance_{i}_{t}"
            )
    
    # 5. 能量上下界
    for i in range(NS):
        for t in range(T + 1):
            model.addConstr(E[i, t] <= E_max * N[i, t], name=f"E_ub_{i}_{t}")
            if t > 0:
                # 出发车辆需要满电
                outflow_t = gp.quicksum(f[i, j, t - 1] for j in range(NS))
                remaining = N[i, t - 1] - outflow_t
                model.addConstr(
                    E[i, t - 1] >= E_max * outflow_t + E_min * remaining,
                    name=f"E_lb_{i}_{t}"
                )
    
    # 6. 充电功率限制
    for i in range(NS):
        for t in range(T):
            model.addConstr(e[i, t] <= e_max * N[i, t], name=f"charge_power_{i}_{t}")
    
    # 7. 周期性条件: 终态 = 初态
    for i in range(NS):
        model.addConstr(N[i, T] == config.N_init, name=f"N_periodic_{i}")
        model.addConstr(E[i, T] >= config.E_init, name=f"E_periodic_{i}")
    
    # --- 求解 ---
    print("\n[Solving...]")
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # --- 提取结果 ---
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        print(f"\n[Solution Found] Status: {model.status}")
        print(f"  Solve Time: {solve_time:.2f} seconds")
        print(f"  Objective Value (Profit): ${model.objVal:,.2f}")
        
        # 提取变量值
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
        
        # 计算定价 c = (a - f) / b
        c_result = np.zeros((NS, NS, T))
        for i in range(NS):
            for j in range(NS):
                for t in range(T):
                    if a[i, j, t] > 0:
                        c_result[i, j, t] = (a[i, j, t] - f_result[i, j, t]) / b
        
        # 计算收益和成本
        total_revenue = 0
        for i in range(NS):
            for j in range(NS):
                for t in range(T):
                    total_revenue += c_result[i, j, t] * f_result[i, j, t]
        
        total_charging_cost = 0
        for i in range(NS):
            for t in range(T):
                bus_id = mapping.get(i, 0)
                total_charging_cost += lmp[bus_id, t] * e_result[i, t]
        
        print(f"  Total Revenue: ${total_revenue:,.2f}")
        print(f"  Total Charging Cost: ${total_charging_cost:,.2f}")
        print()
        
        return {
            'price': c_result,
            'flow': f_result,
            'vehicles': N_result,
            'energy': E_result,
            'charging': e_result,
            'revenue': total_revenue,
            'charging_cost': total_charging_cost,
            'profit': model.objVal,
            'solve_time': solve_time,
            'status': 'optimal' if model.status == GRB.OPTIMAL else 'time_limit'
        }
    else:
        print(f"[Error] Optimization failed with status: {model.status}")
        return None


def solve_miqp_heuristic(config, grid_data, traffic_data):
    """Heuristic fallback (same as before)"""
    # ... (保留原有的启发式实现作为 fallback)
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    b = config.b_elasticity
    
    a = traffic_data.a_demand
    L = traffic_data.L_energy
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    c_price = np.zeros((NS, NS, T))
    f_flow = np.zeros((NS, NS, T))
    N_vehicles = np.zeros((NS, T + 1))
    E_energy = np.zeros((NS, T + 1))
    e_charge = np.zeros((NS, T))
    
    N_vehicles[:, 0] = config.N_init
    E_energy[:, 0] = config.E_init
    
    total_revenue = 0
    total_charging_cost = 0
    
    for t in range(T):
        for i in range(NS):
            for j in range(NS):
                if a[i, j, t] > 0:
                    c_price[i, j, t] = a[i, j, t] / (2 * b)
                    f_flow[i, j, t] = max(0, a[i, j, t] - b * c_price[i, j, t])
        
        for i in range(NS):
            inflow = np.sum(f_flow[:, i, t])
            outflow = np.sum(f_flow[i, :, t])
            N_vehicles[i, t + 1] = N_vehicles[i, t] + inflow - outflow
        
        for i in range(NS):
            energy_out = E_max * np.sum(f_flow[i, :, t])
            energy_in = np.sum((E_max - L[:, i, t]) * f_flow[:, i, t])
            target_energy = E_max * max(N_vehicles[i, t + 1], 0)
            current = E_energy[i, t] - energy_out + energy_in
            e_charge[i, t] = max(0, target_energy - current)
            E_energy[i, t + 1] = current + e_charge[i, t]
        
        for i in range(NS):
            for j in range(NS):
                total_revenue += c_price[i, j, t] * f_flow[i, j, t]
            bus_id = mapping.get(i, 1) - 1
            total_charging_cost += lmp[bus_id, t] * e_charge[i, t]
    
    total_profit = total_revenue - total_charging_cost
    
    return {
        'price': c_price,
        'flow': f_flow,
        'vehicles': N_vehicles,
        'energy': E_energy,
        'charging': e_charge,
        'revenue': total_revenue,
        'charging_cost': total_charging_cost,
        'profit': total_profit,
        'solve_time': 0,
        'status': 'heuristic'
    }


# ============================================================
# 4. ALGORITHM 2: BENDERS DECOMPOSITION (Heuristic Relocation)
# ============================================================

def solve_relocation_subproblem(
    N_vehicles: np.ndarray,
    config: SystemConfig,
    unit_cost: float = 5.0
) -> Tuple[np.ndarray, float]:
    """Most Urgent First 启发式调度"""
    print("=" * 60)
    print("ALGORITHM 2: Relocation Subproblem (MUF Heuristic)")
    print("=" * 60)
    
    NS = config.num_stations
    T = config.T
    target = config.N_init
    base_cap = (config.N_init * NS) * 7.5 # Professional calibration for feasibility
    r_reloc = np.zeros((NS, NS, T))
    total_cost = 0.0
    
    for t in range(T):
        gaps = target - N_vehicles[:, t + 1]
        deficit_stations = [(i, gaps[i]) for i in range(NS) if gaps[i] > 0]
        surplus_stations = [(i, -gaps[i]) for i in range(NS) if gaps[i] < 0]
        
        deficit_stations.sort(key=lambda x: -x[1])
        surplus_stations.sort(key=lambda x: -x[1])
        
        for (d_station, d_need) in deficit_stations:
            for idx, (s_station, s_avail) in enumerate(surplus_stations):
                if s_avail <= 0:
                    continue
                transfer = min(d_need, s_avail)
                r_reloc[s_station, d_station, t] = transfer
                total_cost += transfer * unit_cost
                surplus_stations[idx] = (s_station, s_avail - transfer)
                d_need -= transfer
                if d_need <= 0:
                    break
    
    print(f"  Total Relocation Cost: ${total_cost:,.2f}")
    print()
    
    return r_reloc, total_cost


# ============================================================
# 5. RESULT VISUALIZATION
# ============================================================

def visualize_results(results: Dict, config: SystemConfig, save_path: str = None):
    """可视化实验结果"""
    T = config.T
    NS = config.num_stations
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 车辆分布
    ax1 = axes[0, 0]
    for i in range(NS):
        ax1.plot(range(T + 1), results['vehicles'][i, :], label=f'Station {i + 1}')
    ax1.set_xlabel('Time Period (h)')
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Vehicle Distribution Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 充电曲线
    ax2 = axes[0, 1]
    total_charging = np.sum(results['charging'], axis=0)
    ax2.bar(range(T), total_charging, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Time Period (h)')
    ax2.set_ylabel('Total Charging Energy (kWh)')
    ax2.set_title('Aggregated Charging Profile')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 定价热力图
    ax3 = axes[1, 0]
    price_from_0 = results['price'][0, 1:, :]
    im = ax3.imshow(price_from_0, aspect='auto', cmap='YlOrRd')
    ax3.set_xlabel('Time Period (h)')
    ax3.set_ylabel('Destination Station')
    ax3.set_title('Dynamic Pricing (From Station 1)')
    plt.colorbar(im, ax=ax3, label='Price ($)')
    
    # Plot 4: 利润构成
    ax4 = axes[1, 1]
    labels = ['Revenue', 'Charging Cost', 'Net Profit']
    values = [results['revenue'], results['charging_cost'], results['profit']]
    colors = ['green', 'red', 'blue']
    bars = ax4.bar(labels, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Amount ($)')
    ax4.set_title('Profit Breakdown')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    plt.show()


# ============================================================
# 6. MAIN EXPERIMENT WORKFLOW
# ============================================================

def run_experiment():
    """主实验流程"""
    print("\n" + "=" * 70)
    print("EV SHARING SYSTEM OPTIMIZATION - MAIN EXPERIMENT (GUROBI VERSION)")
    print("=" * 70 + "\n")
    
    # Step 0: 配置
    config = SystemConfig()
    print(f"Configuration: {config.num_stations} stations, {config.T} time periods")
    print(f"Gurobi Available: {GUROBI_AVAILABLE}\n")
    
    # Step 1: 数据准备 (使用真实数据)
    print("[Step 1] Loading real data (IEEE 33 + NYC Taxi)...")
    grid_data, traffic_data = load_real_data(config)
    print("  Done.\n")
    
    # Step 2: 算法 1 - MIQP
    print("[Step 2] Running Algorithm 1 (MIQP)...")
    miqp_results = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    if miqp_results is None:
        print("MIQP failed. Exiting.")
        return None
    
    # Step 3: 算法 2 - Benders (Relocation)
    print("[Step 3] Running Algorithm 2 (Relocation)...")
    r_reloc, reloc_cost = solve_relocation_subproblem(miqp_results['vehicles'], config)
    
    # Step 4: 综合结果
    final_profit = miqp_results['profit'] - reloc_cost
    print("=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  MIQP Solver Status:              {miqp_results['status']}")
    print(f"  MIQP Solve Time:                 {miqp_results['solve_time']:.2f} seconds")
    print(f"  MIQP Profit (before relocation): ${miqp_results['profit']:,.2f}")
    print(f"  Relocation Cost:                 ${reloc_cost:,.2f}")
    print(f"  Final Net Profit:                ${final_profit:,.2f}")
    print()
    
    # Step 5: 可视化
    print("[Step 4] Generating visualizations...")
    visualize_results(miqp_results, config, save_path='experiment_results_gurobi.png')
    
    print("\n[Experiment Completed Successfully]\n")
    
    return miqp_results, r_reloc, reloc_cost


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_experiment()
