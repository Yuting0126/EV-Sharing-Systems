import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gurobipy as gp
from gurobipy import GRB
import json
import time

# Import DES for gap analysis
from des_simulation import EVSharingSimulator

# Import existing modules
from main_experiment_gurobi import (
    SystemConfig, GridData, TrafficData, load_real_data, solve_miqp_gurobi
)
from adp_algorithm import ValueFunction, ADPTrainer
from complete_miqp_lbbd import solve_miqp_complete

OUTPUT_DIR = "1218_result"
# Ensure result directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_adp_learning_curve():
    """Point 6: ADP Learning Curve."""
    print("\n[ADP Learning Curve] Training for 500 episodes...")
    vf = ValueFunction(num_features=25)  # 5 base + 10 stations * 2 (inventory, energy)
    trainer = ADPTrainer(vf)
    
    # Train and get history
    history = trainer.train(num_episodes=500, verbose=True)
    rewards = history['episode_rewards']
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 20
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), ma, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward ($)')
    plt.title('ADP Training Convergence (NYC Taxi Dataset)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/adp_learning_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/adp_learning_curve.png")

def generate_lmp_impact():
    """Point 4: LMP Impact Curves."""
    print("\n[LMP Impact] Running Smart vs Fixed Charging Scenarios...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Scenario 1: Smart Charging (Optimized)
    res_smart = solve_miqp_gurobi(config, grid_data, traffic_data)
    charge_smart = res_smart['charging'].sum(axis=0) # Total hourly charging
    
    # Scenario 2: Fixed Charging (Heuristic)
    # Force charging to follow arrivals immediately
    # We approximate this by making the charging decision e_i,t fixed proportional to arrivals
    # For a fair comparison, we use the same total energy but distributed by arrival time
    total_energy_needed = charge_smart.sum()
    
    # Calculate arrival energy deficit
    arrivals = np.zeros(config.T)
    for t in range(config.T):
        arrivals[t] = traffic_data.a_demand[:, :, t].sum()
    
    if arrivals.sum() > 0:
        charge_fixed = (arrivals / arrivals.sum()) * total_energy_needed
    else:
        charge_fixed = np.full(config.T, total_energy_needed / config.T)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot LMPs on primary axis
    bus_id = 18 # Representative bus
    ax1.plot(grid_data.lmp[bus_id, :], 'k--', alpha=0.5, label=f'LMP (Bus {bus_id})')
    ax1.set_ylabel('Electricity Price ($/kWh)')
    ax1.set_xlabel('Hour of Day')
    
    # Plot Charging on secondary axis
    ax2 = ax1.twinx()
    ax2.fill_between(range(24), charge_fixed, color='red', alpha=0.2, label='Fixed Charging (Baseline)')
    ax2.plot(range(24), charge_smart, color='blue', linewidth=2, marker='o', label='Smart Charging (Proposed)')
    ax2.set_ylabel('Aggregated EV Charging Load (kWh)')
    
    plt.title('Impact of Smart Charging on Grid Load and Energy Costs')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.savefig(f"{OUTPUT_DIR}/lmp_impact.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/lmp_impact.png")

def solve_miqp_epsilon_constrained(config, grid_data, traffic_data, eps_lsr=0.1, cap_factor=1.0):
    """
    Academic Rigor MIQP:
    1. Objective: Maximize Profit
    2. Constraint: LSR <= epsilon
    3. Grid: Aggregate Charging Capacity Constraint (Branch Limit Scaling kappa)
    """
    NS = config.num_stations
    T = config.T
    E_max = config.E_max
    E_min = config.E_min
    b = config.b_elasticity
    N_max = config.N_max
    e_max_per = config.e_max_per_period
    
    a = traffic_data.a_demand
    L = traffic_data.L_energy
    lmp = grid_data.lmp
    mapping = traffic_data.station_to_bus
    
    model = gp.Model("EV_Epsilon_Constrained")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 30
    
    f = model.addVars(NS, NS, T, lb=0, name="f")
    N = model.addVars(NS, T + 1, lb=0, ub=N_max, name="N")
    E = model.addVars(NS, T + 1, lb=0, name="E")
    e = model.addVars(NS, T, lb=0, name="e")
    r = model.addVars(NS, NS, T, lb=0, name="r")
    
    # Costs
    gamma = 5.0 # Unit relocation cost
    revenue = gp.quicksum((a[i,j,t]*f[i,j,t] - f[i,j,t]**2)/b for i in range(NS) for j in range(NS) for t in range(T) if a[i,j,t]>0)
    # Planning-based charging cost: total charging scheduled (e[i,t]) with 5x LMP amplifier
    # This is a planning cost - represents preventive charging to maintain fleet feasibility
    charging_cost = gp.quicksum(5.0 * lmp[mapping.get(i,0),t]*e[i,t] for i in range(NS) for t in range(T))
    reloc_cost = gp.quicksum(gamma * r[i,k,t] for i in range(NS) for k in range(NS) for t in range(T))
    
    model.setObjective(revenue - charging_cost - reloc_cost, GRB.MAXIMIZE)
    
    # Operational Constraints
    for i in range(NS):
        model.addConstr(N[i,0] == config.N_init)
        model.addConstr(E[i,0] == config.E_init)
        
        # Demand upper bound: f <= a
        for j in range(NS):
            for t in range(T):
                if a[i,j,t] > 0:
                    model.addConstr(f[i,j,t] <= a[i,j,t])
                else:
                    model.addConstr(f[i,j,t] == 0)

        for t in range(T):
            # Vehicle flow: include relocation
            inflow = gp.quicksum(f[j,i,t] for j in range(NS)) + gp.quicksum(r[k,i,t] for k in range(NS))
            outflow = gp.quicksum(f[i,j,t] for j in range(NS)) + gp.quicksum(r[i,k,t] for k in range(NS))
            model.addConstr(N[i,t+1] == N[i,t] + inflow - outflow)
            
            # Energy flow: include relocation (assume relocation follows same energy policy)
            en_in = gp.quicksum((E_max - L[j,i,t]) * f[j,i,t] for j in range(NS)) + \
                    gp.quicksum((E_max - L[k,i,t]) * r[k,i,t] for k in range(NS)) # Simplified relocation energy
            en_out = gp.quicksum(E_max * f[i,j,t] for j in range(NS)) + \
                     gp.quicksum(E_max * r[i,k,t] for k in range(NS))
            model.addConstr(E[i,t+1] == E[i,t] + e[i,t] - en_out + en_in)
            
            model.addConstr(E[i,t] <= E_max * N[i,t])
            model.addConstr(e[i,t] <= e_max_per * N[i,t])
            # Relaxed departure constraint: allow half-charge departure for feasibility
            total_outflow = gp.quicksum(f[i,j,t] for j in range(NS)) + gp.quicksum(r[i,k,t] for k in range(NS))
            model.addConstr(E[i,t] >= 0.5 * E_max * total_outflow + E_min * (N[i,t] - total_outflow))
            
        # Relaxed terminal constraints for feasibility across demand scenarios
        model.addConstr(N[i,T] >= config.N_init * 0.5)  # Allow 50% vehicle deficit
        model.addConstr(E[i,T] >= config.E_init * 0.3)  # Allow 70% energy deficit

    # Epsilon-Constraint: LSR <= eps_lsr
    total_a = a.sum()
    served = gp.quicksum(f[i,j,t] for i in range(NS) for j in range(NS) for t in range(T) if a[i,j,t]>0)
    model.addConstr((total_a - served) <= eps_lsr * total_a, name="lsr_limit")

    # Grid Tightness (kappa): Branch Line Capacity Scaling
    # Grid Tightness (kappa): Branch Limit Scaling
    # Tighter multiplier to force different frontiers per kappa
    base_cap = (config.N_init * NS) * 2.0  # ~16,000 kW, more binding 
    for t in range(T):
        model.addConstr(e.sum('*', t) <= base_cap * cap_factor, name=f"grid_cap_{t}")

    model.optimize()
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Optimization result
        served_val = served.getValue()
        f_res = np.zeros((NS, NS, T))
        prices = np.zeros((NS, NS, T))
        for i in range(NS):
            for j in range(NS):
                for t in range(T):
                    f_val = f[i,j,t].X
                    f_res[i,j,t] = f_val
                    if a[i,j,t] > 0:
                        prices[i,j,t] = (a[i,j,t] - f_val) / b
        
        # Energy Balance Verification (for audit)
        total_charged = sum(e[i,t].X for i in range(NS) for t in range(T))
        served_trip_energy = sum(L[i,j,t] * f[i,j,t].X for i in range(NS) for j in range(NS) for t in range(T))
        reloc_energy = sum(L[i,j,t] * r[i,j,t].X for i in range(NS) for j in range(NS) for t in range(T))
        initial_soc = sum(E[i,0].X for i in range(NS))
        final_soc = sum(E[i,T].X for i in range(NS))
        soc_change = final_soc - initial_soc
        energy_balance_error = total_charged - served_trip_energy - reloc_energy - soc_change
        
        return {
            'profit': model.objVal,
            'revenue': revenue.getValue(),
            'charging_cost_total': charging_cost.getValue(),
            'reloc_cost': reloc_cost.getValue(),
            'lsr': (1 - served_val/total_a)*100,
            'service_rate': (served_val/total_a)*100,
            'peak_load': max(sum(e[i,t].X for i in range(NS)) for t in range(T)),
            'prices': prices,
            # Energy balance audit
            'total_charged_kwh': total_charged,
            'served_trip_energy_kwh': served_trip_energy,
            'reloc_energy_kwh': reloc_energy,
            'soc_change_kwh': soc_change,
            'energy_balance_error_kwh': energy_balance_error
        }
    return None

def generate_pareto_frontier():
    """Point 7: Professional Epsilon-Constraint Pareto Frontier (V8)."""
    print("\n[Pareto Frontier] Running Epsilon-Constraint Overhaul...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Unified Fleet Scale: 800 per station
    config.N_init = 800
    config.E_init = 800 * 21 * 0.7
    config.N_max = 800 * 2
    
    # Grid Tightness (kappa) scales: adjusted for feasibility
    kappas = [1.5, 1.3, 1.1, 1.0, 0.9]
    # Epsilon thresholds (LSR targets): 15 points
    epsilons = np.linspace(0.01, 0.25, 15) # From 1% to 25% LSR
    
    series_results = []
    for kappa in kappas:
        print(f"  Testing Grid Tightness kappa={kappa}...")
        points = []
        for eps in epsilons:
            res = solve_miqp_epsilon_constrained(config, grid_data, traffic_data, eps_lsr=eps, cap_factor=kappa)
            if res:
                points.append(res)
        
        # Identify non-dominated points for this series
        if points:
            # Sort by Service Rate ascending for the plot line
            points.sort(key=lambda x: x['service_rate'])
            # Log exact values for debugging
            sr_vals = [f"{p['service_rate']:.1f}" for p in points]
            pr_vals = [f"{p['profit']:.0f}" for p in points]
            print(f"    -> Kappa {kappa}: SR={sr_vals}")
            print(f"    -> Kappa {kappa}: PR={pr_vals}")
            series_results.append({'kappa': kappa, 'all': points, 'frontier': points})

    # Plotting
    plt.figure(figsize=(11, 7))
    colors = ['#27ae60', '#2980b9', '#f39c12', '#e67e22', '#c0392b']
    
    # We want a unified colorbar for Peak Load
    all_peak_loads = []
    for res in series_results:
        all_peak_loads.extend([p['peak_load'] for p in res['all']])
    
    # Check for negative profit and warn
    all_profits = []
    for res in series_results:
        all_profits.extend([p['profit'] for p in res['all']])
    if any(p < 0 for p in all_profits):
        print(f"  [Warning] Detected negative profits (min={min(all_profits):.1f}). This implies operational losses at high SR targets.")

    vmin_load = 0
    vmax_load = max(all_peak_loads) if all_peak_loads else 100
    
    sc_mappable = None
    # Define marker shapes for each κ (grid tightness) - matches current kappas
    shape_map = {1.5: 'o', 1.3: 's', 1.1: '^', 1.0: 'D', 0.9: '*'}
    for idx, (res, color) in enumerate(zip(series_results, colors)):
        all_pts = res['all']
        frontier = res['frontier']
        
        srv_all = [p['service_rate'] for p in all_pts]
        prf_all = [p['profit'] for p in all_pts]
        pk_all = [p['peak_load'] for p in all_pts]
        
        srv_f = [p['service_rate'] for p in frontier]
        prf_f = [p['profit'] for p in frontier]
        
        label = f'$\kappa$={res["kappa"]}'
        
        # Draw line
        plt.plot(srv_f, prf_f, '-', color=color, linewidth=2.5, alpha=0.9, zorder=4)
        # Scatter for Peak Load colormap
        sc = plt.scatter(srv_all, prf_all, c=pk_all, cmap='viridis', s=60,
                        marker=shape_map.get(res['kappa'], 'o'), edgecolors='white', linewidth=0.5, alpha=0.8,
                        vmin=vmin_load, vmax=vmax_load, zorder=3)
        sc_mappable = sc
        # Markers for frontier
        plt.scatter(srv_f, prf_f, facecolors='none', edgecolors=color, s=100, 
                   linewidth=1.5, label=label, zorder=5)

    plt.xlabel('Service Rate (1 - LSR) %', fontsize=12)
    plt.ylabel('Daily Profit ($)', fontsize=12)
    plt.title('Operational Trilemma: Profit vs. Service Rate under Distribution Constraints', fontsize=14, fontweight='bold')
    
    if sc_mappable:
        cbar = plt.colorbar(sc_mappable)
        cbar.set_label('Peak Grid Load (kW)', fontsize=11)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.savefig(f"{OUTPUT_DIR}/pareto_frontier.png", dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/pareto_frontier.png")

def generate_trilemma_corner_analysis():
    """Perspective B: Policy Corner Analysis (User vs. Grid vs. Operator)."""
    print("\n[Trilemma Corner Analysis] Running Policy Scenarios...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Consistent Scale: 8000 vehicles total (800 per zone)
    BASE_FLEET = 800
    config.N_init = BASE_FLEET
    config.E_init = BASE_FLEET * 21
    config.N_max = BASE_FLEET * 2
    
    # Define Corners
    results_raw = {}
    
    # 1. Profit-Maximizer (Balanced)
    print("  Solving Profit-Maximizer...")
    results_raw['Profit'] = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    # 2. User-Centric (Maximize Service Level)
    # We simulate this by increasing fleet size significiantly (1500 per station)
    print("  Solving User-Centric...")
    config.N_init = 1500
    config.E_init = 1500 * 21
    config.N_max = 1500 * 2
    results_raw['User'] = solve_miqp_gurobi(config, grid_data, traffic_data)
    
    # 3. Grid-Friendly (Minimize Peak Load)
    # Restore base fleet, but double LMP and cap charging power or just use the price signal
    print("  Solving Grid-Friendly...")
    config.N_init = BASE_FLEET
    config.E_init = BASE_FLEET * 21
    config.N_max = BASE_FLEET * 2
    temp_lmp = grid_data.lmp * 2.0 # More reasonable scaling than 5x
    temp_grid = GridData(grid_data.bus_df, grid_data.branch_df, grid_data.base_load, temp_lmp)
    # Also simulate power capping by setting a lower charging limit if possible, 
    # but scaling LMP is the most direct 'endogenous' way in this model.
    results_raw['Grid'] = solve_miqp_gurobi(config, temp_grid, traffic_data)
    
    # Metric Extraction
    metrics = {}
    total_potential = traffic_data.a_demand.sum()
    
    for name, res in results_raw.items():
        if res is None:
            print(f"  [Warning] Scaling for {name} failed. Using defaults.")
            metrics[name] = [0, 100, 100] # Worst case
            continue
            
        served = res['flow'].sum()
        profit = res['profit'] / 1e3
        lsr = ((total_potential - served) / total_potential) * 100
        # Peak Loading (max aggregate charging)
        peak_load = res['charging'].sum(axis=0).max()
        metrics[name] = [profit, lsr, peak_load]

    # Normalize for Radar Chart (0-100 scale)
    # Profit: scale by ~1200
    # Service: 100 - LSR
    # Grid: Max peak is around 8000, invert to Friendliness
    def norm_radar(p, l, g):
        # High value = Good
        return [
            min(100, (p / 1200.0) * 100), # Profitability
            100 - l,                      # Service Level
            max(0, 100 - (g / 150.0) * 100) # Grid Friendliness (inverted peak)
        ]

    v_profit = norm_radar(*metrics['Profit'])
    v_user = norm_radar(*metrics['User'])
    v_grid = norm_radar(*metrics['Grid'])
    
    # Radar Plotting
    # (Reuse radar_factory logic or just simple bar/radar)
    labels = np.array(['Profitability', 'Service Level', 'Grid Friendliness'])
    theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    
    # Close the loop
    v_profit = np.concatenate((v_profit, [v_profit[0]]))
    v_user = np.concatenate((v_user, [v_user[0]]))
    v_grid = np.concatenate((v_grid, [v_grid[0]]))
    theta_close = np.concatenate((theta, [theta[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(theta_close, v_profit, 'o-', color='blue', label='Profit-Maximizer', linewidth=2)
    ax.fill(theta_close, v_profit, alpha=0.2, color='blue')
    
    ax.plot(theta_close, v_user, 's-', color='green', label='User-Centric (High Service)', linewidth=2)
    ax.fill(theta_close, v_user, alpha=0.2, color='green')
    
    ax.plot(theta_close, v_grid, '^-', color='red', label='Grid-Friendly (Low Impact)', linewidth=2)
    ax.fill(theta_close, v_grid, alpha=0.2, color='red')
    
    ax.set_thetagrids(np.degrees(theta), labels)
    ax.set_ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('The Operational Trilemma: Policy Corner Analysis', y=1.05, fontsize=14, fontweight='bold')
    
    plt.savefig(f"{OUTPUT_DIR}/trilemma_triangle.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/trilemma_triangle.png")

def generate_implementation_gap():
    """Point 8: Implementation Gap Analysis (Ideal vs. Realized)."""
    print("\n[Implementation Gap] Running MIQP vs. DES Comparison...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Unified Fleet Scale: 800 per station
    config.N_init = 800
    config.E_init = 800 * 21 * 0.7
    config.N_max = 800 * 2
    
    od_matrix = np.load("data/processed/od_matrix_week.npy")[:, :, :, 0] # Day 1
    
    # Select more epsilon points for kappa=1.0 - Use feasible range (higher eps = lower target SR)
    epsilons = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    
    miqp_pts = []
    des_pts = []
    
    for eps in epsilons:
        print(f"  Testing Implementation for eps={eps}...")
        # Relax constraints (cap_factor=15.0) to match DES's unconstrained charging environment
        # and see the GAP caused by stochasticity rather than different constraint models.
        res = solve_miqp_epsilon_constrained(config, grid_data, traffic_data, eps_lsr=eps, cap_factor=15.0)
        if res:
            miqp_sr = res['service_rate']
            miqp_profit = res['profit']  # This should be: Revenue - Charging Cost - Relocation Cost
            target_sr = (1 - eps) * 100
            miqp_pts.append({'sr': target_sr, 'profit': miqp_profit})
            print(f"    [MIQP eps={eps}] SR={miqp_sr:.1f}%, Profit=${miqp_profit:,.0f}")
            
            # Run DES with these prices
            sim = EVSharingSimulator(
                num_stations=config.num_stations,
                vehicles_per_station=config.N_init,
                od_matrix=od_matrix,
                energy_matrix=traffic_data.L_energy,
                external_prices=res['prices'],
                b_elasticity=config.b_elasticity,
                seed=42
            )
            # Alignment: match the base charging rate to MIQP's average LMP (~$0.20)
            sim.charging_cost_rate = 0.20
            sim_res = sim.run(duration_minutes=24*60)
            
            # Detailed DES diagnostics & Aligned Calculations
            des_served = sim_res['total_trips']
            des_lost = sim_res['lost_demand']
            des_rev = sim_res['total_revenue']
            # Align profit: applying the same 5x amplifier used in MIQP to the charging cost
            des_charging_aligned = sim_res['charging_cost'] * 5.0
            des_reloc_cost = sim_res['relocation_cost']
            des_profit_aligned = des_rev - des_charging_aligned - des_reloc_cost
            
            # Align Service Rate: Use MIQP's 'total_a' (potential demand) as denominator
            # instead of DES's realized arrivals, ensuring identical metric basis.
            des_sr_aligned = (des_served / np.sum(traffic_data.a_demand)) * 100
            
            print(f"    [DES  eps={eps}] Aligned SR={des_sr_aligned:.1f}%, Aligned Profit=${des_profit_aligned:,.0f}")
            print(f"    [DES  eps={eps}] Raw Served={des_served}, Raw Lost={des_lost}")
            
            # Store metrics for plotting
            # X-axis: Use Target SR (1-eps) for BOTH lines, making comparison at same target
            # Y-axis: Aligned profit (Rev - Charge*5 - Reloc)
            target_sr = (1 - eps) * 100
            des_pts.append({'sr': target_sr, 'profit': des_profit_aligned, 'realized_sr': des_sr_aligned})
            
            # Additional Breakdown for Alignment Verification
            print(f"    [Breakdown MIQP] Rev=${res.get('revenue',0):,.0f}, Charge=${res.get('charging_cost_total',0):,.0f}, Reloc=${res.get('reloc_cost',0):,.0f}")
            print(f"    [Breakdown DES ] Rev=${des_rev:,.0f}, Charge=${des_charging_aligned:,.0f}, Reloc=${des_reloc_cost:,.0f}")
            # Energy Balance Audit for MIQP
            print(f"    [MIQP Energy] Charged={res.get('total_charged_kwh',0):,.0f}kWh, TripE={res.get('served_trip_energy_kwh',0):,.0f}kWh, RelocE={res.get('reloc_energy_kwh',0):,.0f}kWh, ΔSoC={res.get('soc_change_kwh',0):,.0f}kWh, Error={res.get('energy_balance_error_kwh',0):,.1f}kWh")
        else:
            print(f"    [MIQP eps={eps}] INFEASIBLE - skipped")

    # Plotting
    plt.figure(figsize=(9, 6))
    miqp_sr = [p['sr'] for p in miqp_pts]
    miqp_pr = [p['profit'] for p in miqp_pts]
    des_sr = [p['sr'] for p in des_pts]
    des_pr = [p['profit'] for p in des_pts]
    
    print(f"  [Diagnostics Final] Target SRs: {[p['sr'] for p in miqp_pts]}")
    print(f"  [Diagnostics Final] DES Realized SRs: {[p.get('realized_sr', 'N/A') for p in des_pts]}")
    print(f"  [Diagnostics Final] MIQP Profits: {miqp_pr}")
    print(f"  [Diagnostics Final] DES  Profits: {des_pr}")
    
    plt.plot(miqp_sr, miqp_pr, 'b--o', label='Planned (MIQP)', alpha=0.6, markersize=10)
    plt.plot(des_sr, des_pr, 'r-s', label='Realized (DES)', linewidth=2, markersize=10)
    
    # Draw arrows representing the gap
    for i in range(len(miqp_sr)):
        plt.annotate('', xy=(des_sr[i], des_pr[i]), xytext=(miqp_sr[i], miqp_pr[i]),
                     arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

    plt.xlabel('Target Service Rate (1 - ε) %', fontsize=12)
    plt.ylabel('Daily Net Profit ($)', fontsize=12)
    plt.title('Implementation Gap: Planned (MIQP) vs. Realized (DES) at Same Target')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/implementation_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/implementation_gap.png")


def generate_implementation_gap_mc(K=10):
    """
    Monte Carlo version of Implementation Gap Analysis for Figure 6.
    Runs K replications for each epsilon point with CRN.
    Reports mean ± 95% CI for all metrics.
    """
    print(f"\n[Monte Carlo Implementation Gap] K={K} replications per epsilon...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Unified Fleet Scale: 800 per station
    config.N_init = 800
    config.E_init = 800 * 21 * 0.7
    config.N_max = 800 * 2
    
    od_matrix = np.load("data/processed/od_matrix_week.npy")[:, :, :, 0]
    epsilons = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    
    all_results = []
    
    for eps in epsilons:
        print(f"\n  Testing eps={eps} with K={K} replications...")
        target_sr = (1 - eps) * 100
        
        # Run MIQP once (deterministic)
        res = solve_miqp_epsilon_constrained(config, grid_data, traffic_data, eps_lsr=eps, cap_factor=15.0)
        if not res:
            print(f"    [MIQP eps={eps}] INFEASIBLE - skipped")
            continue
        
        miqp_profit = res['profit']
        miqp_sr = res['service_rate']
        print(f"    [MIQP eps={eps}] SR={miqp_sr:.1f}%, Profit=${miqp_profit:,.0f}")
        
        # Run K DES replications with CRN
        replication_data = []
        for k in range(1, K + 1):
            sim = EVSharingSimulator(
                num_stations=config.num_stations,
                vehicles_per_station=config.N_init,
                od_matrix=od_matrix,
                energy_matrix=traffic_data.L_energy,
                external_prices=res['prices'],
                b_elasticity=config.b_elasticity,
                seed=k  # CRN
            )
            sim.charging_cost_rate = 0.20
            sim_res = sim.run(duration_minutes=24*60)
            
            des_served = sim_res['total_trips']
            des_lost = sim_res['lost_demand']
            des_rev = sim_res['total_revenue']
            des_charge = sim_res['charging_cost'] * 5.0
            des_reloc = sim_res['relocation_cost']
            des_profit = des_rev - des_charge - des_reloc
            
            des_sr_aligned = (des_served / np.sum(traffic_data.a_demand)) * 100
            gap_pct = ((des_profit - miqp_profit) / abs(miqp_profit)) * 100 if miqp_profit != 0 else 0
            abs_gap = miqp_profit - des_profit
            
            replication_data.append({
                'des_profit': des_profit,
                'gap_pct': gap_pct,
                'abs_gap': abs_gap,
                'des_sr_aligned': des_sr_aligned
            })
            
            if k % 5 == 0 or k == K:
                print(f"    Rep {k}/{K}: Profit=${des_profit:,.0f}, Gap={gap_pct:.1f}%")
        
        # Calculate statistics
        def calc_stats(data_list, key):
            values = np.array([d[key] for d in data_list])
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            ci_half = 1.96 * std / np.sqrt(len(values))
            return mean, ci_half, std
        
        stats = {}
        for key in ['des_profit', 'gap_pct', 'abs_gap', 'des_sr_aligned']:
            mean, ci, std = calc_stats(replication_data, key)
            stats[key] = {'mean': mean, 'ci': ci, 'std': std}
        
        print(f"\n    [MC Summary] eps={eps} (Target SR={target_sr:.0f}%)")
        print(f"      DES Profit: ${stats['des_profit']['mean']:,.0f} ± ${stats['des_profit']['ci']:,.0f}")
        print(f"      Gap: {stats['gap_pct']['mean']:.2f}% ± {stats['gap_pct']['ci']:.2f}%")
        
        all_results.append({
            'eps': eps,
            'target_sr': target_sr,
            'miqp_profit': miqp_profit,
            'miqp_sr': miqp_sr,
            'K': K,
            'stats': stats
        })
    
    # Plot with error bars
    plt.figure(figsize=(9, 6))
    
    # MIQP line (deterministic)
    miqp_sr_vals = [r['target_sr'] for r in all_results]
    miqp_profit_vals = [r['miqp_profit'] for r in all_results]
    
    # DES with error bars
    des_sr_vals = [r['target_sr'] for r in all_results]
    des_profit_means = [r['stats']['des_profit']['mean'] for r in all_results]
    des_profit_cis = [r['stats']['des_profit']['ci'] for r in all_results]
    
    plt.plot(miqp_sr_vals, miqp_profit_vals, 'b--o', label='Planned (MIQP)', alpha=0.6, markersize=10)
    plt.errorbar(des_sr_vals, des_profit_means, yerr=des_profit_cis, fmt='r-s', 
                 label=f'Realized (DES, mean ± 95% CI, K={K})', linewidth=2, markersize=10,
                 capsize=5, capthick=2)
    
    plt.xlabel('Target Service Rate (1 - ε) %', fontsize=12)
    plt.ylabel('Daily Net Profit ($)', fontsize=12)
    plt.title(f'Implementation Gap (K={K} Monte Carlo replications)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/implementation_gap_mc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/implementation_gap_mc.png")
    
    return all_results


def generate_uncertainty_sweep():
    """
    Stress test: Vary demand intensity to show how planning-realization gap changes.
    At high demand volatility, MIQP's conservative charging should become advantageous.
    """
    print("\n[Uncertainty Sweep] Running Demand Intensity Stress Test...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    # Unified Fleet Scale: 800 per station (MATCHING FIG 6)
    config.N_init = 800
    config.E_init = 800 * 21 * 0.7
    config.N_max = config.N_init * 2
    
    # Demand intensity multipliers (0.8x to 1.3x)
    intensities = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    eps = 0.35  # Set to 0.35 (Target 65%) to match the mid-range of Figure 6
    
    results = []
    
    for intensity in intensities:
        print(f"  Testing intensity={intensity:.2f}x with eps={eps}...")
        
        # Scale demand
        scaled_demand = traffic_data.a_demand * intensity
        scaled_traffic = TrafficData(
            a_demand=scaled_demand,
            L_energy=traffic_data.L_energy,
            station_to_bus=traffic_data.station_to_bus
        )
        
        # Run MIQP - use cap_factor=15.0 consistent with Fig 6
        res = solve_miqp_epsilon_constrained(config, grid_data, scaled_traffic, eps_lsr=eps, cap_factor=15.0)
        if not res:
            print(f"    MIQP infeasible at intensity={intensity}")
            continue
        
        miqp_profit = res['profit']
        
        # Run DES
        od_matrix = np.load("data/processed/od_matrix_week.npy")[:, :, :, 0] * intensity
        sim = EVSharingSimulator(
            num_stations=config.num_stations,
            vehicles_per_station=config.N_init,
            od_matrix=od_matrix,
            energy_matrix=traffic_data.L_energy,
            external_prices=res['prices'],
            b_elasticity=config.b_elasticity,
            seed=42
        )
        sim.charging_cost_rate = 0.20
        sim_res = sim.run(duration_minutes=24*60)
        
        des_rev = sim_res['total_revenue']
        des_charge = sim_res['charging_cost'] * 5.0
        des_reloc = sim_res['relocation_cost']
        des_profit = des_rev - des_charge - des_reloc
        
        gap = des_profit - miqp_profit
        gap_pct = (gap / abs(miqp_profit)) * 100 if miqp_profit != 0 else 0
        
        # === Aligned Evidence Metrics ===
        d_pot_total = np.sum(scaled_demand)
        des_served = sim_res['total_trips']
        des_lost = sim_res['lost_demand']
        
        sr_realized = (des_served / (des_served + des_lost)) * 100 if (des_served + des_lost) > 0 else 0
        sr_aligned = (des_served / d_pot_total) * 100 if d_pot_total > 0 else 0
        miqp_sr = res.get('service_rate', 0.0)
        eta = (sr_aligned / miqp_sr) * 100 if miqp_sr > 0 else 0
        
        miqp_rev = res.get('revenue', 0.0)
        miqp_charge = res.get('charging_cost_total', 0.0)
        miqp_reloc = res.get('reloc_cost', 0.0)
        
        abs_gap = miqp_profit - des_profit
        delta_rev = miqp_rev - des_rev
        delta_charge = miqp_charge - des_charge
        delta_reloc = miqp_reloc - des_reloc
        
        print(f"    [Aligned Evidence] Int={intensity:.1f}x")
        print(f"      SR: MIQP={miqp_sr:.1f}% | DES_aligned={sr_aligned:.1f}% | DES_realized={sr_realized:.1f}%")
        print(f"      Execution Efficiency η = {eta:.1f}%")
        print(f"      Gap: Abs=${abs_gap:,.0f} | Rel={gap_pct:+.1f}%")
        print(f"      Δ Breakdown: ΔRev=${delta_rev:,.0f} | ΔCharge=${delta_charge:,.0f} | ΔReloc=${delta_reloc:,.0f}")
        
        results.append({
            'intensity': intensity,
            'miqp_profit': miqp_profit,
            'des_profit': des_profit,
            'gap': gap,
            'gap_pct': gap_pct,
            'abs_gap': abs_gap,
            'miqp_sr': miqp_sr,
            'sr_aligned': sr_aligned,
            'sr_realized': sr_realized,
            'eta': eta,
            'delta_rev': delta_rev,
            'delta_charge': delta_charge,
            'delta_reloc': delta_reloc
        })
    
    # Plot
    plt.figure(figsize=(10, 6))
    intensities_plot = [r['intensity'] for r in results]
    gaps_plot = [r['gap_pct'] for r in results]
    
    plt.plot(intensities_plot, gaps_plot, 'b-o', markersize=10, linewidth=2, label='Profit Gap')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.fill_between(intensities_plot, gaps_plot, 0, alpha=0.2, color='red')
    
    for i, (x, y) in enumerate(zip(intensities_plot, gaps_plot)):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    plt.xlabel('Demand Intensity Multiplier', fontsize=12)
    plt.ylabel('Profit Gap: (DES − MIQP) / MIQP  (%)', fontsize=12)
    plt.title('Planning Advantage vs. Demand Intensity')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/uncertainty_sweep.png")
    
    return results


def generate_uncertainty_sweep_mc(K=20, intensities=None):
    """
    Monte Carlo version of uncertainty sweep.
    Runs K replications for each demand intensity with Common Random Numbers (CRN).
    Reports mean ± 95% CI for all metrics.
    """
    print(f"\n[Monte Carlo Uncertainty Sweep] K={K} replications per point...")
    config = SystemConfig()
    grid_data, traffic_data = load_real_data(config)
    
    config.N_init = 800
    config.E_init = 800 * 21 * 0.7
    config.N_max = config.N_init * 2
    
    if intensities is None:
        intensities = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    eps = 0.35
    
    all_results = []
    
    for intensity in intensities:
        print(f"\n  Testing intensity={intensity:.2f}x with K={K} replications...")
        
        scaled_demand = traffic_data.a_demand * intensity
        scaled_traffic = TrafficData(
            a_demand=scaled_demand,
            L_energy=traffic_data.L_energy,
            station_to_bus=traffic_data.station_to_bus
        )
        
        res = solve_miqp_epsilon_constrained(config, grid_data, scaled_traffic, eps_lsr=eps, cap_factor=15.0)
        if not res:
            print(f"    MIQP infeasible at intensity={intensity}")
            continue
        
        miqp_profit = res['profit']
        miqp_sr = res.get('service_rate', 0.0)
        miqp_rev = res.get('revenue', 0.0)
        miqp_charge = res.get('charging_cost_total', 0.0)
        miqp_reloc = res.get('reloc_cost', 0.0)
        d_pot_total = np.sum(scaled_demand)
        
        replication_data = []
        od_matrix = np.load("data/processed/od_matrix_week.npy")[:, :, :, 0] * intensity
        
        for k in range(1, K + 1):
            sim = EVSharingSimulator(
                num_stations=config.num_stations,
                vehicles_per_station=config.N_init,
                od_matrix=od_matrix,
                energy_matrix=traffic_data.L_energy,
                external_prices=res['prices'],
                b_elasticity=config.b_elasticity,
                seed=k
            )
            sim.charging_cost_rate = 0.20
            sim_res = sim.run(duration_minutes=24*60)
            
            des_rev = sim_res['total_revenue']
            des_charge = sim_res['charging_cost'] * 5.0
            des_reloc = sim_res['relocation_cost']
            des_profit = des_rev - des_charge - des_reloc
            des_served = sim_res['total_trips']
            des_lost = sim_res['lost_demand']
            
            sr_realized = (des_served / (des_served + des_lost)) * 100 if (des_served + des_lost) > 0 else 0
            sr_aligned = (des_served / d_pot_total) * 100 if d_pot_total > 0 else 0
            eta = (sr_aligned / miqp_sr) * 100 if miqp_sr > 0 else 0
            
            gap_pct = ((des_profit - miqp_profit) / abs(miqp_profit)) * 100 if miqp_profit != 0 else 0
            abs_gap = miqp_profit - des_profit
            delta_rev = miqp_rev - des_rev
            delta_charge = miqp_charge - des_charge
            delta_reloc = miqp_reloc - des_reloc
            
            replication_data.append({
                'des_profit': des_profit,
                'gap_pct': gap_pct,
                'abs_gap': abs_gap,
                'sr_aligned': sr_aligned,
                'sr_realized': sr_realized,
                'eta': eta,
                'delta_rev': delta_rev,
                'delta_charge': delta_charge,
                'delta_reloc': delta_reloc
            })
            
            if k % 5 == 0 or k == K:
                print(f"    Replication {k}/{K} done. Profit=${des_profit:,.0f}, Gap={gap_pct:.1f}%")
        
        def calc_stats(data_list, key):
            values = np.array([d[key] for d in data_list])
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            ci_half = 1.96 * std / np.sqrt(len(values))
            return mean, ci_half, std
        
        stats = {}
        for key in ['des_profit', 'gap_pct', 'abs_gap', 'sr_aligned', 'sr_realized', 'eta', 
                    'delta_rev', 'delta_charge', 'delta_reloc']:
            mean, ci, std = calc_stats(replication_data, key)
            stats[key] = {'mean': mean, 'ci': ci, 'std': std}
        
        print(f"\n    [MC Summary] Intensity={intensity:.1f}x (K={K})")
        print(f"      DES Profit: ${stats['des_profit']['mean']:,.0f} ± ${stats['des_profit']['ci']:,.0f}")
        print(f"      Rel Gap: {stats['gap_pct']['mean']:.2f}% ± {stats['gap_pct']['ci']:.2f}%")
        print(f"      Abs Gap: ${stats['abs_gap']['mean']:,.0f} ± ${stats['abs_gap']['ci']:,.0f}")
        print(f"      SR_aligned: {stats['sr_aligned']['mean']:.2f}% ± {stats['sr_aligned']['ci']:.2f}%")
        print(f"      η: {stats['eta']['mean']:.1f}% ± {stats['eta']['ci']:.1f}%")
        
        all_results.append({
            'intensity': intensity,
            'miqp_profit': miqp_profit,
            'miqp_sr': miqp_sr,
            'K': K,
            'stats': stats
        })
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    x_vals = [r['intensity'] for r in all_results]
    y_means = [r['stats']['gap_pct']['mean'] for r in all_results]
    y_cis = [r['stats']['gap_pct']['ci'] for r in all_results]
    
    plt.errorbar(x_vals, y_means, yerr=y_cis, fmt='b-o', markersize=10, linewidth=2, 
                 capsize=5, capthick=2, label=f'Profit Gap (mean ± 95% CI, K={K})')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.fill_between(x_vals, [y - c for y, c in zip(y_means, y_cis)], 
                     [y + c for y, c in zip(y_means, y_cis)], alpha=0.15, color='blue')
    
    for x, y, c in zip(x_vals, y_means, y_cis):
        plt.annotate(f'{y:.1f}%±{c:.1f}', (x, y), textcoords="offset points", 
                     xytext=(0, 12), ha='center', fontsize=8)
    
    plt.xlabel('Demand Intensity Multiplier', fontsize=12)
    plt.ylabel('Profit Gap: (DES − MIQP) / MIQP  (%)', fontsize=12)
    plt.title(f'Planning Advantage vs. Demand Intensity (K={K} Monte Carlo replications)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/uncertainty_sweep_mc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/uncertainty_sweep_mc.png")
    
    return all_results


if __name__ == "__main__":

    # Ensure result directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    generate_lmp_impact()
    generate_pareto_frontier()
    generate_trilemma_corner_analysis()
    generate_implementation_gap()
