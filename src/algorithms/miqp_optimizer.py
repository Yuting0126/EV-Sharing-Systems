"""
Integrated MIQP + LBBD 7-Day Experiment

This script runs a complete 7-day simulation integrating:
- Algorithm 1 (MIQP): Daily pricing and charging optimization
- Algorithm 2 (LBBD): Nightly vehicle relocation optimization

Comparison:
- With Relocation: MIQP daily + LBBD nightly
- Without Relocation: MIQP daily only, imbalance accumulates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import gurobipy as gp
from gurobipy import GRB
import os
import json
from datetime import datetime

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "1218_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Unified configuration for integrated experiment."""
    num_stations = 10
    num_hours = 24
    vehicles_per_station = 800
    E_max = 70.0  # kWh max battery
    E_min = 10.0  # kWh min battery
    charging_rate = 7.0  # kW
    price_elasticity = 5.0
    relocation_cost = 5.0  # $ per vehicle
    lost_demand_penalty = 15.0  # $ per lost demand


# =============================================================================
# Algorithm 1: MIQP Daily Optimization
# =============================================================================

def run_miqp_day(config: Config, 
                 od_demand: np.ndarray, 
                 N_init: np.ndarray, 
                 E_init: np.ndarray,
                 lmp: np.ndarray = None,
                 verbose: bool = False) -> dict:
    """
    Run MIQP optimization for one day (24 hours).
    
    Args:
        config: Configuration object
        od_demand: OD demand matrix (NS, NS, 24)
        N_init: Initial vehicle counts per station (NS,)
        E_init: Initial energy per station (NS,)
        lmp: Electricity prices (optional)
        verbose: Print Gurobi output
        
    Returns:
        dict with profit, end_state, served_demand, lost_demand
    """
    NS = config.num_stations
    T = config.num_hours
    
    # Default LMP
    if lmp is None:
        lmp = np.ones((NS, T)) * 0.1  # $0.1/kWh
    
    try:
        model = gp.Model("MIQP_Day")
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.TimeLimit = 300  # 5 min timeout
        model.Params.MIPGap = 0.05  # 5% gap acceptable
        
        # Decision variables
        # f[i,j,t]: number of trips from i to j at time t
        f = {}
        for i in range(NS):
            for j in range(NS):
                for t in range(T):
                    if od_demand[i, j, t] > 0:
                        f[i, j, t] = model.addVar(lb=0, ub=od_demand[i, j, t], name=f"f_{i}_{j}_{t}")
        
        # N[i,t]: vehicles at station i at time t
        N = {}
        for i in range(NS):
            for t in range(T + 1):
                N[i, t] = model.addVar(lb=0, ub=config.vehicles_per_station * 3, name=f"N_{i}_{t}")
        
        # E[i,t]: total energy at station i at time t
        E = {}
        for i in range(NS):
            for t in range(T + 1):
                E[i, t] = model.addVar(lb=0, name=f"E_{i}_{t}")
        
        # e[i,t]: charging amount at station i at time t
        e = {}
        for i in range(NS):
            for t in range(T):
                e[i, t] = model.addVar(lb=0, ub=config.vehicles_per_station * config.charging_rate, name=f"e_{i}_{t}")
        
        model.update()
        
        # Objective: Revenue - Charging Cost
        revenue = gp.quicksum(
            od_demand[i, j, t] * f[i, j, t] - f[i, j, t] * f[i, j, t] / config.price_elasticity
            for i in range(NS) for j in range(NS) for t in range(T)
            if (i, j, t) in f
        )
        
        charging_cost = gp.quicksum(
            lmp[i % lmp.shape[0], t] * e[i, t]
            for i in range(NS) for t in range(T)
        )
        
        model.setObjective(revenue - charging_cost, GRB.MAXIMIZE)
        
        # Constraints
        # Initial conditions
        for i in range(NS):
            model.addConstr(N[i, 0] == N_init[i], name=f"N_init_{i}")
            model.addConstr(E[i, 0] == E_init[i], name=f"E_init_{i}")
        
        # Vehicle balance: N[i,t+1] = N[i,t] - outflow + inflow
        for i in range(NS):
            for t in range(T):
                outflow = gp.quicksum(f[i, j, t] for j in range(NS) if (i, j, t) in f)
                inflow = gp.quicksum(f[j, i, t] for j in range(NS) if (j, i, t) in f)
                model.addConstr(N[i, t+1] == N[i, t] - outflow + inflow, name=f"N_balance_{i}_{t}")
        
        # Flow constraint: outflow <= available vehicles
        for i in range(NS):
            for t in range(T):
                outflow = gp.quicksum(f[i, j, t] for j in range(NS) if (i, j, t) in f)
                model.addConstr(outflow <= N[i, t], name=f"outflow_{i}_{t}")
        
        # Energy balance (simplified - remove strict bounds)
        energy_per_trip = 10.0  # kWh average
        for i in range(NS):
            for t in range(T):
                usage = gp.quicksum(energy_per_trip * f[i, j, t] for j in range(NS) if (i, j, t) in f)
                # Relaxed energy balance with charging
                model.addConstr(E[i, t+1] >= E[i, t] - usage, name=f"E_balance_{i}_{t}")
                model.addConstr(E[i, t+1] <= E[i, t] - usage + e[i, t], name=f"E_charge_{i}_{t}")
        
        # Simplified energy bounds (only ensure non-negative)
        for i in range(NS):
            for t in range(T):
                model.addConstr(E[i, t] >= 0, name=f"E_min_{i}_{t}")
        
        # Solve
        model.optimize()
        
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            # Extract results
            profit = model.ObjVal
            
            # End-of-day state
            N_end = np.array([N[i, T].X for i in range(NS)])
            E_end = np.array([E[i, T].X for i in range(NS)])
            
            # Calculate served and lost demand
            served = sum(f[i, j, t].X for (i, j, t) in f)
            total_demand = od_demand.sum()
            lost = total_demand - served
            
            return {
                'status': 'optimal',
                'profit': profit,
                'N_end': N_end,
                'E_end': E_end,
                'served_demand': served,
                'lost_demand': lost,
                'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0
            }
        else:
            return {
                'status': 'infeasible',
                'profit': 0,
                'N_end': N_init.copy(),
                'E_end': E_init.copy(),
                'served_demand': 0,
                'lost_demand': od_demand.sum()
            }
            
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return {
            'status': f'error: {e}',
            'profit': 0,
            'N_end': N_init.copy(),
            'E_end': E_init.copy(),
            'served_demand': 0,
            'lost_demand': od_demand.sum()
        }


# =============================================================================
# Algorithm 2: LBBD Relocation (Simplified for Integration)
# =============================================================================

def run_lbbd_relocation(config: Config,
                        N_current: np.ndarray,
                        tomorrow_demand: np.ndarray) -> dict:
    """
    Run LBBD to decide overnight relocation.
    
    Simplified version: rebalance to average distribution.
    
    Args:
        config: Configuration object
        N_current: Current vehicle distribution
        tomorrow_demand: Expected demand for tomorrow (NS, NS, 24)
        
    Returns:
        dict with relocations, cost, N_after
    """
    NS = config.num_stations
    
    # Calculate target based on tomorrow's outgoing demand
    outgoing_demand = tomorrow_demand.sum(axis=1).sum(axis=1)  # (NS,) total outgoing per station
    total_vehicles = N_current.sum()
    
    # Target: proportional to demand, but capped at current total
    if outgoing_demand.sum() > 0:
        target = outgoing_demand / outgoing_demand.sum() * total_vehicles
    else:
        target = np.full(NS, total_vehicles / NS)
    
    # Ensure minimum vehicles per station
    target = np.maximum(target, 50)
    target = target / target.sum() * total_vehicles  # Normalize
    
    # Calculate relocations needed
    surplus = np.maximum(N_current - target, 0)
    deficit = np.maximum(target - N_current, 0)
    
    relocations = []
    reloc_cost = 0.0
    
    # Simple greedy matching
    surplus_stations = np.argsort(-surplus)
    deficit_stations = np.argsort(-deficit)
    
    N_after = N_current.copy()
    
    for src in surplus_stations:
        if surplus[src] <= 0:
            continue
        for dst in deficit_stations:
            if deficit[dst] <= 0:
                continue
            
            to_move = min(surplus[src], deficit[dst])
            if to_move > 0:
                relocations.append((src, dst, int(to_move)))
                reloc_cost += to_move * config.relocation_cost
                
                N_after[src] -= to_move
                N_after[dst] += to_move
                
                surplus[src] -= to_move
                deficit[dst] -= to_move
    
    return {
        'relocations': relocations,
        'cost': reloc_cost,
        'N_after': N_after,
        'vehicles_moved': sum(r[2] for r in relocations)
    }


# =============================================================================
# Integrated 7-Day Experiment
# =============================================================================

def run_integrated_experiment(num_days: int = 7, verbose: bool = True):
    """
    Run complete 7-day experiment with MIQP + LBBD integration.
    
    Compares:
    - With LBBD Relocation: MIQP daily + LBBD nightly rebalancing
    - Without Relocation: MIQP daily, no rebalancing
    """
    config = Config()
    NS = config.num_stations
    
    print("="*70)
    print("INTEGRATED MIQP + LBBD 7-DAY EXPERIMENT")
    print("="*70)
    print(f"Stations: {NS}")
    print(f"Vehicles/Station: {config.vehicles_per_station}")
    print(f"Total Vehicles: {NS * config.vehicles_per_station}")
    print(f"Days: {num_days}")
    
    # Load OD data
    od_path = "data/processed/od_matrix_week.npy"
    if os.path.exists(od_path):
        od_week = np.load(od_path)  # (10, 10, 24, 7)
        print(f"Loaded OD data: {od_week.shape}")
    else:
        print("ERROR: OD data not found!")
        return None
    
    # Initial state (balanced)
    N_init = np.full(NS, config.vehicles_per_station, dtype=float)
    E_init = N_init * config.E_max * 0.7  # 70% charged
    
    # Track results
    results_with_reloc = []
    results_no_reloc = []
    
    # Current states (separate for each scenario)
    N_with = N_init.copy()
    E_with = E_init.copy()
    N_no = N_init.copy()
    E_no = E_init.copy()
    
    print("\n" + "="*70)
    print("RUNNING SIMULATIONS...")
    print("="*70)
    
    for day in range(num_days):
        day_idx = day % 7  # Cycle through week
        od_today = od_week[:, :, :, day_idx]
        od_tomorrow = od_week[:, :, :, (day_idx + 1) % 7]
        
        day_name = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'][day_idx]
        print(f"\n--- Day {day+1} ({day_name}) | Demand: {od_today.sum():,.0f} trips ---")
        
        # === Scenario 1: With LBBD Relocation ===
        print(f"  [MIQP + LBBD] Running optimization...")
        miqp_result_with = run_miqp_day(config, od_today, N_with, E_with, verbose=False)
        
        if miqp_result_with['status'] == 'optimal':
            # Run LBBD for overnight relocation
            lbbd_result = run_lbbd_relocation(config, miqp_result_with['N_end'], od_tomorrow)
            
            daily_profit_with = miqp_result_with['profit'] - lbbd_result['cost']
            N_with = lbbd_result['N_after']
            # Overnight charging: replenish energy to 70%
            E_with = N_with * config.E_max * 0.7
            
            results_with_reloc.append({
                'day': day + 1,
                'profit': miqp_result_with['profit'],
                'reloc_cost': lbbd_result['cost'],
                'net_profit': daily_profit_with,
                'served': miqp_result_with['served_demand'],
                'lost': miqp_result_with['lost_demand'],
                'vehicles_moved': lbbd_result['vehicles_moved']
            })
            
            print(f"    Profit: ${miqp_result_with['profit']:,.0f}, Reloc: ${lbbd_result['cost']:,.0f}, "
                  f"Net: ${daily_profit_with:,.0f}, Moved: {lbbd_result['vehicles_moved']}")
        else:
            print(f"    Status: {miqp_result_with['status']}")
            results_with_reloc.append({'day': day+1, 'profit': 0, 'reloc_cost': 0, 'net_profit': 0, 'served': 0, 'lost': od_today.sum(), 'vehicles_moved': 0})
        
        # === Scenario 2: Without Relocation ===
        print(f"  [MIQP only] Running optimization...")
        miqp_result_no = run_miqp_day(config, od_today, N_no, E_no, verbose=False)
        
        if miqp_result_no['status'] == 'optimal':
            N_no = miqp_result_no['N_end']  # State accumulates (no relocation)
            # Overnight charging: replenish energy to 70%
            E_no = N_no * config.E_max * 0.7
            
            results_no_reloc.append({
                'day': day + 1,
                'profit': miqp_result_no['profit'],
                'reloc_cost': 0,
                'net_profit': miqp_result_no['profit'],
                'served': miqp_result_no['served_demand'],
                'lost': miqp_result_no['lost_demand'],
                'imbalance': np.std(N_no)
            })
            
            print(f"    Profit: ${miqp_result_no['profit']:,.0f}, Imbalance: {np.std(N_no):.1f}")
        else:
            print(f"    Status: {miqp_result_no['status']}")
            results_no_reloc.append({'day': day+1, 'profit': 0, 'net_profit': 0, 'served': 0, 'lost': od_today.sum(), 'imbalance': np.std(N_no)})
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    df_with = pd.DataFrame(results_with_reloc)
    df_no = pd.DataFrame(results_no_reloc)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    total_profit_with = df_with['net_profit'].sum()
    total_profit_no = df_no['net_profit'].sum()
    total_reloc_cost = df_with['reloc_cost'].sum()
    
    print(f"\nWith LBBD Relocation:")
    print(f"  Total Gross Profit: ${df_with['profit'].sum():,.0f}")
    print(f"  Total Relocation Cost: ${total_reloc_cost:,.0f}")
    print(f"  Total Net Profit: ${total_profit_with:,.0f}")
    print(f"  Total Lost Demand: {df_with['lost'].sum():,.0f}")
    
    print(f"\nWithout Relocation:")
    print(f"  Total Profit: ${total_profit_no:,.0f}")
    print(f"  Total Lost Demand: {df_no['lost'].sum():,.0f}")
    print(f"  Final Imbalance: {df_no['imbalance'].iloc[-1]:.1f}")
    
    advantage = total_profit_with - total_profit_no
    pct = advantage / abs(total_profit_no) * 100 if total_profit_no != 0 else 0
    
    print(f"\n** LBBD Relocation Advantage: ${advantage:,.0f} ({pct:+.1f}%) **")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    # Save CSV
    df_with.to_csv(f"{OUTPUT_DIR}/integrated_with_reloc.csv", index=False)
    df_no.to_csv(f"{OUTPUT_DIR}/integrated_no_reloc.csv", index=False)
    print(f"\nSaved CSV results to {OUTPUT_DIR}/")
    
    # Create charts
    create_integrated_charts(df_with, df_no, advantage, pct)
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_days': num_days,
            'vehicles_per_station': config.vehicles_per_station,
            'num_stations': config.num_stations
        },
        'with_relocation': {
            'total_net_profit': total_profit_with,
            'total_reloc_cost': total_reloc_cost,
            'total_lost_demand': df_with['lost'].sum()
        },
        'no_relocation': {
            'total_profit': total_profit_no,
            'total_lost_demand': df_no['lost'].sum(),
            'final_imbalance': float(df_no['imbalance'].iloc[-1])
        },
        'advantage': advantage,
        'advantage_pct': pct
    }
    
    with open(f"{OUTPUT_DIR}/integrated_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return df_with, df_no


def create_integrated_charts(df_with, df_no, advantage, pct):
    """Create comparison charts for integrated experiment."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    # 1. Cumulative Net Profit
    ax1 = axes[0, 0]
    days = df_with['day']
    ax1.plot(days, df_with['net_profit'].cumsum()/1e3, 'o-', color=colors[0], 
             linewidth=2, markersize=6, label='MIQP + LBBD')
    ax1.plot(days, df_no['net_profit'].cumsum()/1e3, 's--', color=colors[1], 
             linewidth=2, markersize=6, label='MIQP Only')
    ax1.fill_between(days, df_no['net_profit'].cumsum()/1e3, df_with['net_profit'].cumsum()/1e3, 
                     alpha=0.2, color=colors[0])
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Cumulative Net Profit ($K)')
    ax1.set_title('Cumulative Profit Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Net Profit
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(days))
    ax2.bar(x - width/2, df_with['net_profit']/1e3, width, label='MIQP + LBBD', color=colors[0])
    ax2.bar(x + width/2, df_no['net_profit']/1e3, width, label='MIQP Only', color=colors[1])
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Daily Net Profit ($K)')
    ax2.set_title('Daily Profit Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Day {d}' for d in days])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lost Demand Comparison
    ax3 = axes[1, 0]
    ax3.plot(days, df_with['lost'], 'o-', color=colors[0], linewidth=2, markersize=6, label='MIQP + LBBD')
    ax3.plot(days, df_no['lost'], 's--', color=colors[1], linewidth=2, markersize=6, label='MIQP Only')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Lost Demand (trips)')
    ax3.set_title('Lost Demand Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Bar Chart
    ax4 = axes[1, 1]
    metrics = ['Total Profit\n($K)', 'Lost Demand\n(×1000)', 'Reloc Cost\n($K)']
    with_vals = [
        df_with['profit'].sum()/1e3, 
        df_with['lost'].sum()/1e3, 
        df_with['reloc_cost'].sum()/1e3
    ]
    no_vals = [
        df_no['profit'].sum()/1e3, 
        df_no['lost'].sum()/1e3, 
        0
    ]
    
    x = np.arange(len(metrics))
    ax4.bar(x - width/2, with_vals, width, label='MIQP + LBBD', color=colors[0])
    ax4.bar(x + width/2, no_vals, width, label='MIQP Only', color=colors[1])
    ax4.set_ylabel('Value')
    ax4.set_title(f'Summary Comparison | Advantage: ${advantage/1e3:,.0f}K ({pct:+.1f}%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/integrated_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved integrated_comparison.png")


if __name__ == "__main__":
    run_integrated_experiment(num_days=7, verbose=True)
