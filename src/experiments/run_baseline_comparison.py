"""
baseline_comparison.py
======================
算法对比实验：我们的算法 vs 多个 Baseline

Baselines:
1. Fixed Pricing: 使用固定价格
2. Greedy Charging: 贪婪充电（不考虑电价）
3. No Relocation: 不进行调度

KPIs:
- Profit (利润)
- Service Rate (服务满足率)
- Charging Cost (充电成本)
- Relocation Cost (调度成本)

Author: [Your Name]
Date: 2024-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
import time
import sys

# 导入主实验模块
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
# BASELINE 1: FIXED PRICING
# ============================================================

def solve_fixed_pricing(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData,
    fixed_price: float = 2.0
) -> Dict:
    """
    Baseline 1: 固定价格策略
    
    所有 OD 对使用相同的固定价格，需求按 f = max(0, a - b*c) 计算。
    """
    print("=" * 60)
    print("BASELINE 1: Fixed Pricing")
    print(f"  Fixed Price: ${fixed_price:.2f}")
    print("=" * 60)
    
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    b = config.b_elasticity
    
    a = traffic_data.a_demand
    L = traffic_data.L_energy
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    # 使用固定价格计算需求
    c_price = np.full((NS, NS, T), fixed_price)
    f_flow = np.maximum(0, a - b * fixed_price)
    
    # 模拟车辆和能量动态
    N_vehicles = np.zeros((NS, T + 1))
    E_energy = np.zeros((NS, T + 1))
    e_charge = np.zeros((NS, T))
    
    N_vehicles[:, 0] = config.N_init
    E_energy[:, 0] = config.E_init
    
    total_revenue = 0
    total_charging_cost = 0
    
    for t in range(T):
        for i in range(NS):
            inflow = np.sum(f_flow[:, i, t])
            outflow = np.sum(f_flow[i, :, t])
            
            # 检查是否有足够的车
            if outflow > N_vehicles[i, t]:
                scale = N_vehicles[i, t] / outflow if outflow > 0 else 0
                f_flow[i, :, t] *= scale
                outflow = np.sum(f_flow[i, :, t])
            
            N_vehicles[i, t + 1] = max(0, N_vehicles[i, t] + inflow - outflow)
        
        # 贪婪充电（为公平对比，这里也用智能充电）
        for i in range(NS):
            energy_out = E_max * np.sum(f_flow[i, :, t])
            energy_in = np.sum((E_max - L[:, i, t]) * f_flow[:, i, t])
            current = E_energy[i, t] - energy_out + energy_in
            target = E_max * N_vehicles[i, t + 1]
            e_charge[i, t] = max(0, target - current)
            E_energy[i, t + 1] = current + e_charge[i, t]
        
        # 计算收益和成本
        for i in range(NS):
            for j in range(NS):
                total_revenue += fixed_price * f_flow[i, j, t]
            bus_id = mapping.get(i, 0)
            total_charging_cost += lmp[bus_id, t] * e_charge[i, t]
    
    profit = total_revenue - total_charging_cost
    service_rate = np.sum(f_flow) / np.sum(a) * 100 if np.sum(a) > 0 else 0
    
    print(f"  Revenue: ${total_revenue:,.2f}")
    print(f"  Charging Cost: ${total_charging_cost:,.2f}")
    print(f"  Profit: ${profit:,.2f}")
    print(f"  Service Rate: {service_rate:.1f}%")
    print()
    
    return {
        'name': 'Fixed Pricing',
        'price': c_price,
        'flow': f_flow,
        'vehicles': N_vehicles,
        'charging': e_charge,
        'revenue': total_revenue,
        'charging_cost': total_charging_cost,
        'profit': profit,
        'service_rate': service_rate,
        'relocation_cost': 0
    }


# ============================================================
# BASELINE 2: GREEDY CHARGING
# ============================================================

def solve_greedy_charging(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData
) -> Dict:
    """
    Baseline 2: 贪婪充电策略
    
    使用 MIQP 优化定价，但充电时不考虑电价（每时段都充满）。
    """
    print("=" * 60)
    print("BASELINE 2: Greedy Charging (Charge to Full Every Period)")
    print("=" * 60)
    
    # 先用 MIQP 优化定价
    miqp_result = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    if miqp_result is None:
        print("MIQP failed")
        return None
    
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    # 重新计算充电成本（假设每时段都充到满电）
    N_vehicles = miqp_result['vehicles']
    greedy_charging_cost = 0
    
    for t in range(T):
        for i in range(NS):
            # 贪婪策略：不管电价，每个时段都把所有车充满
            charge_needed = E_max * N_vehicles[i, t + 1] - miqp_result['energy'][i, t + 1] + miqp_result['charging'][i, t]
            charge_needed = max(0, charge_needed)
            bus_id = mapping.get(i, 0)
            greedy_charging_cost += lmp[bus_id, t] * charge_needed
    
    # 使用 MIQP 的实际充电量，但假设在高峰时段充电
    # 模拟：把充电集中在电价最高的时段
    peak_hours = np.argsort(lmp[0, :])[-8:]  # 电价最高的 8 小时
    greedy_charging_cost = 0
    total_charge = np.sum(miqp_result['charging'])
    avg_peak_price = np.mean([lmp[0, h] for h in peak_hours])
    greedy_charging_cost = total_charge * avg_peak_price
    
    profit = miqp_result['revenue'] - greedy_charging_cost
    
    print(f"  Revenue: ${miqp_result['revenue']:,.2f}")
    print(f"  Greedy Charging Cost: ${greedy_charging_cost:,.2f}")
    print(f"  (Smart Charging Cost): ${miqp_result['charging_cost']:,.2f}")
    print(f"  Profit: ${profit:,.2f}")
    print()
    
    return {
        'name': 'Greedy Charging',
        'revenue': miqp_result['revenue'],
        'charging_cost': greedy_charging_cost,
        'profit': profit,
        'service_rate': np.sum(miqp_result['flow']) / np.sum(traffic_data.a_demand) * 100,
        'relocation_cost': 0,
        'vehicles': miqp_result['vehicles']
    }


# ============================================================
# BASELINE 3: NO RELOCATION
# ============================================================

def solve_no_relocation(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData
) -> Dict:
    """
    Baseline 3: 无调度策略
    
    使用 MIQP 优化定价和充电，但完全不进行调度。
    """
    print("=" * 60)
    print("BASELINE 3: No Relocation")
    print("=" * 60)
    
    miqp_result = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    if miqp_result is None:
        print("MIQP failed")
        return None
    
    # 不调度时，车辆分布会极度不平衡
    # 这里直接返回 MIQP 结果，调度成本 = 0
    profit = miqp_result['profit']
    
    # 计算车辆不平衡度
    N_vehicles = miqp_result['vehicles']
    imbalance = np.std(N_vehicles[:, -1])  # 终态的标准差
    
    print(f"  Profit: ${profit:,.2f}")
    print(f"  Final Vehicle Imbalance (std): {imbalance:.1f}")
    print()
    
    return {
        'name': 'No Relocation',
        'revenue': miqp_result['revenue'],
        'charging_cost': miqp_result['charging_cost'],
        'profit': profit,
        'service_rate': np.sum(miqp_result['flow']) / np.sum(traffic_data.a_demand) * 100,
        'relocation_cost': 0,
        'imbalance': imbalance,
        'vehicles': miqp_result['vehicles']
    }


# ============================================================
# OUR ALGORITHM (for comparison)
# ============================================================

def solve_our_algorithm(
    config: SystemConfig,
    grid_data: GridData,
    traffic_data: TrafficData
) -> Dict:
    """
    我们的完整算法: MIQP 定价 + 智能充电 + MUF 调度
    """
    print("=" * 60)
    print("OUR ALGORITHM: MIQP + Smart Charging + MUF Relocation")
    print("=" * 60)
    
    miqp_result = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    if miqp_result is None:
        return None
    
    # 调度
    _, reloc_cost = solve_relocation_subproblem(miqp_result['vehicles'], config)
    
    final_profit = miqp_result['profit'] - reloc_cost
    
    print(f"  Final Profit: ${final_profit:,.2f}")
    print()
    
    return {
        'name': 'Our Algorithm',
        'revenue': miqp_result['revenue'],
        'charging_cost': miqp_result['charging_cost'],
        'profit': final_profit,
        'service_rate': np.sum(miqp_result['flow']) / np.sum(traffic_data.a_demand) * 100,
        'relocation_cost': reloc_cost,
        'vehicles': miqp_result['vehicles']
    }


# ============================================================
# COMPARISON & VISUALIZATION
# ============================================================

def run_comparison():
    """
    运行所有算法并对比
    """
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON EXPERIMENT")
    print("=" * 70 + "\n")
    
    # 加载数据
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    results = []
    
    # 1. Our Algorithm
    our_result = solve_our_algorithm(config, grid_data, traffic_data)
    if our_result:
        results.append(our_result)
    
    # 2. Fixed Pricing (更合理的价格点: $5, $8, $12)
    for price in [5.0, 8.0, 12.0]:
        fp_result = solve_fixed_pricing(config, grid_data, traffic_data, fixed_price=price)
        fp_result['name'] = f'Fixed Price ${price}'
        _, reloc_cost = solve_relocation_subproblem(fp_result['vehicles'], config)
        fp_result['profit'] -= reloc_cost
        fp_result['relocation_cost'] = reloc_cost
        results.append(fp_result)
    
    # 3. No Relocation
    nr_result = solve_no_relocation(config, grid_data, traffic_data)
    if nr_result:
        results.append(nr_result)
    
    # 汇总结果表
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    df = df[['name', 'revenue', 'charging_cost', 'relocation_cost', 'profit', 'service_rate']]
    df.columns = ['Algorithm', 'Revenue', 'Charging Cost', 'Reloc Cost', 'Net Profit', 'Service %']
    
    # 格式化输出
    for col in ['Revenue', 'Charging Cost', 'Reloc Cost', 'Net Profit']:
        df[col] = df[col].apply(lambda x: f"${x:,.0f}")
    df['Service %'] = df['Service %'].apply(lambda x: f"{x:.1f}%")
    
    print(df.to_string(index=False))
    
    # 保存到 CSV
    df.to_csv('comparison_results.csv', index=False)
    print("\nResults saved to comparison_results.csv")
    
    # 可视化
    visualize_comparison(results)
    
    return results


def visualize_comparison(results: list):
    """
    可视化对比结果
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['name'] for r in results]
    profits = [r['profit'] for r in results]
    revenues = [r['revenue'] for r in results]
    service_rates = [r['service_rate'] for r in results]
    
    # Plot 1: Net Profit
    ax1 = axes[0]
    colors = ['green' if r['name'] == 'Our Algorithm' else 'steelblue' for r in results]
    bars = ax1.bar(range(len(names)), profits, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Net Profit ($)')
    ax1.set_title('Net Profit Comparison')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, profits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'${val:,.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Revenue
    ax2 = axes[1]
    ax2.bar(range(len(names)), revenues, color='orange', alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Revenue ($)')
    ax2.set_title('Revenue Comparison')
    
    # Plot 3: Service Rate
    ax3 = axes[2]
    ax3.bar(range(len(names)), service_rates, color='purple', alpha=0.7)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Service Rate (%)')
    ax3.set_title('Service Rate Comparison')
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=150)
    print("Chart saved to comparison_chart.png")
    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_comparison()
