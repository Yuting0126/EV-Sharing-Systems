"""
Algorithm 2: Logic-Based Benders Decomposition (LBBD) for Vehicle Relocation

This implements the LBBD approach from the original paper for optimal vehicle relocation.
The existing MUF (Most Urgent First) implementation is preserved in multi_day_simulation.py.

LBBD Structure:
- Master Problem: Integer program deciding relocation routes (which vehicles to move where)
- Slave Problem: Check feasibility of serving demand given relocations
- Benders Cuts: Add cuts when slave is infeasible to guide master

Reference: Xie et al. (2020) - EV Sharing Optimization
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB


@dataclass
class RelocationPlan:
    """Result of LBBD optimization."""
    relocations: List[Tuple[int, int, int]]  # (from, to, count)
    total_cost: float
    iterations: int
    gap: float
    status: str


@dataclass
class LBBDConfig:
    """Configuration for LBBD algorithm."""
    num_stations: int = 10
    max_iterations: int = 50
    time_limit: float = 300.0  # seconds
    gap_tolerance: float = 0.01
    relocation_cost: float = 5.0  # $ per vehicle relocated
    lost_demand_penalty: float = 15.0  # $ per lost trip
    vehicles_per_station: int = 300


class LBBDRelocationSolver:
    """
    Logic-Based Benders Decomposition solver for vehicle relocation.
    
    The problem is decomposed into:
    1. Master: Decide vehicle relocations (IP)
    2. Slave: Check if demand can be served (LP feasibility)
    
    Benders cuts are added when the slave problem indicates infeasibility.
    """
    
    def __init__(self, config: LBBDConfig = None):
        self.config = config or LBBDConfig()
        self.NS = self.config.num_stations
        
        # Store cuts for warm starts
        self.cuts: List[Dict] = []
        self.iteration = 0
        
    def solve(self, 
              vehicle_counts: np.ndarray,
              od_demand: np.ndarray,
              verbose: bool = True) -> RelocationPlan:
        """
        Solve the vehicle relocation problem using LBBD.
        
        Args:
            vehicle_counts: Current vehicles at each station (NS,)
            od_demand: Expected OD demand matrix (NS, NS)
            verbose: Print progress
            
        Returns:
            RelocationPlan with optimal relocations
        """
        if verbose:
            print(f"\n{'='*60}")
            print("LBBD VEHICLE RELOCATION SOLVER")
            print(f"{'='*60}")
            print(f"Stations: {self.NS}")
            print(f"Total vehicles: {vehicle_counts.sum():.0f}")
            print(f"Total demand: {od_demand.sum():.0f}")
        
        # Initialize bounds
        lower_bound = 0.0
        upper_bound = float('inf')
        best_solution = None
        
        self.iteration = 0
        
        while self.iteration < self.config.max_iterations:
            self.iteration += 1
            
            if verbose:
                print(f"\n--- Iteration {self.iteration} ---")
            
            # Step 1: Solve Master Problem
            master_result = self._solve_master(vehicle_counts, od_demand)
            
            if master_result['status'] != 'optimal':
                if verbose:
                    print(f"Master problem status: {master_result['status']}")
                break
            
            relocations = master_result['relocations']
            master_cost = master_result['cost']
            
            # Update lower bound
            lower_bound = max(lower_bound, master_cost)
            
            if verbose:
                print(f"Master cost: ${master_cost:.2f}")
                print(f"Relocations: {len(relocations)}")
            
            # Step 2: Solve Slave Problem (check feasibility)
            slave_result = self._solve_slave(
                vehicle_counts, od_demand, relocations
            )
            
            slave_cost = slave_result['cost']
            total_cost = master_cost + slave_cost
            
            if verbose:
                print(f"Slave cost (lost demand): ${slave_cost:.2f}")
                print(f"Total cost: ${total_cost:.2f}")
            
            # Update upper bound
            if total_cost < upper_bound:
                upper_bound = total_cost
                best_solution = relocations.copy()
                
                if verbose:
                    print(f"* New best solution found!")
            
            # Check convergence
            gap = (upper_bound - lower_bound) / max(upper_bound, 1e-6)
            
            if verbose:
                print(f"Gap: {gap*100:.2f}%")
            
            if gap <= self.config.gap_tolerance:
                if verbose:
                    print("Converged!")
                break
            
            # Step 3: Add Benders Cut if needed
            if slave_result['infeasible_stations']:
                cut = self._generate_benders_cut(
                    vehicle_counts, od_demand, 
                    relocations, slave_result['infeasible_stations']
                )
                self.cuts.append(cut)
                
                if verbose:
                    print(f"Added Benders cut for stations: {slave_result['infeasible_stations']}")
        
        # Prepare result
        final_relocations = []
        if best_solution:
            for (i, j), count in best_solution.items():
                if count > 0:
                    final_relocations.append((i, j, int(count)))
        
        result = RelocationPlan(
            relocations=final_relocations,
            total_cost=upper_bound if upper_bound < float('inf') else 0.0,
            iterations=self.iteration,
            gap=gap if 'gap' in dir() else 0.0,
            status='optimal' if gap <= self.config.gap_tolerance else 'feasible'
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("LBBD RESULT")
            print(f"{'='*60}")
            print(f"Status: {result.status}")
            print(f"Iterations: {result.iterations}")
            print(f"Total cost: ${result.total_cost:.2f}")
            print(f"Relocations: {len(result.relocations)}")
            for r in result.relocations[:5]:
                print(f"  Station {r[0]} -> Station {r[1]}: {r[2]} vehicles")
            if len(result.relocations) > 5:
                print(f"  ... and {len(result.relocations)-5} more")
        
        return result
    
    def _solve_master(self, 
                      vehicle_counts: np.ndarray, 
                      od_demand: np.ndarray) -> Dict:
        """
        Solve the Master Problem: minimize relocation cost subject to flow conservation.
        
        Decision variables:
            r[i,j] = number of vehicles to relocate from i to j
        """
        try:
            model = gp.Model("LBBD_Master")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = self.config.time_limit / 2
            
            NS = self.NS
            
            # Decision variables: relocation count
            r = {}
            for i in range(NS):
                for j in range(NS):
                    if i != j:
                        max_relocate = int(vehicle_counts[i])
                        r[i, j] = model.addVar(
                            vtype=GRB.INTEGER, 
                            lb=0, 
                            ub=max_relocate,
                            name=f"r_{i}_{j}"
                        )
            
            # Slack variables for infeasibility
            slack = {}
            for i in range(NS):
                slack[i] = model.addVar(lb=0, name=f"slack_{i}")
            
            model.update()
            
            # Objective: minimize relocation cost + infeasibility penalty
            relocation_cost = gp.quicksum(
                self.config.relocation_cost * r[i, j]
                for i in range(NS) for j in range(NS) if i != j
            )
            
            penalty = gp.quicksum(
                self.config.lost_demand_penalty * slack[i]
                for i in range(NS)
            )
            
            model.setObjective(relocation_cost + penalty, GRB.MINIMIZE)
            
            # Constraint: Net outflow <= available vehicles
            for i in range(NS):
                outflow = gp.quicksum(r[i, j] for j in range(NS) if j != i)
                inflow = gp.quicksum(r[j, i] for j in range(NS) if j != i)
                
                model.addConstr(
                    outflow - inflow <= vehicle_counts[i],
                    name=f"flow_conservation_{i}"
                )
            
            # Constraint: Demand coverage (relaxed with slack)
            for i in range(NS):
                outgoing_demand = od_demand[i, :].sum()
                vehicles_after = (
                    vehicle_counts[i] 
                    - gp.quicksum(r[i, j] for j in range(NS) if j != i)
                    + gp.quicksum(r[j, i] for j in range(NS) if j != i)
                )
                
                model.addConstr(
                    vehicles_after + slack[i] >= outgoing_demand,
                    name=f"demand_coverage_{i}"
                )
            
            # Add previous Benders cuts
            for cut_idx, cut in enumerate(self.cuts):
                stations = cut['stations']
                min_vehicles = cut['min_vehicles']
                
                for s in stations:
                    vehicles_at_s = (
                        vehicle_counts[s]
                        - gp.quicksum(r[s, j] for j in range(NS) if j != s)
                        + gp.quicksum(r[j, s] for j in range(NS) if j != s)
                    )
                    model.addConstr(
                        vehicles_at_s >= min_vehicles,
                        name=f"benders_cut_{cut_idx}_{s}"
                    )
            
            # Solve
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                relocations = {}
                for i in range(NS):
                    for j in range(NS):
                        if i != j:
                            val = r[i, j].X
                            if val > 0.5:
                                relocations[(i, j)] = round(val)
                
                return {
                    'status': 'optimal',
                    'cost': model.ObjVal,
                    'relocations': relocations
                }
            else:
                return {
                    'status': 'infeasible',
                    'cost': float('inf'),
                    'relocations': {}
                }
                
        except gp.GurobiError as e:
            return {
                'status': f'error: {e}',
                'cost': float('inf'),
                'relocations': {}
            }
    
    def _solve_slave(self,
                     vehicle_counts: np.ndarray,
                     od_demand: np.ndarray,
                     relocations: Dict) -> Dict:
        """
        Solve the Slave Problem: check if demand can be served given relocations.
        
        Returns cost of unserved demand and list of infeasible stations.
        """
        NS = self.NS
        
        # Calculate vehicles after relocation
        vehicles_after = vehicle_counts.copy().astype(float)
        
        for (i, j), count in relocations.items():
            vehicles_after[i] -= count
            vehicles_after[j] += count
        
        # Check each station
        total_lost = 0.0
        infeasible_stations = []
        
        for i in range(NS):
            outgoing_demand = od_demand[i, :].sum()
            available = vehicles_after[i]
            
            if outgoing_demand > available:
                lost = outgoing_demand - available
                total_lost += lost
                infeasible_stations.append(i)
        
        cost = total_lost * self.config.lost_demand_penalty
        
        return {
            'cost': cost,
            'lost_demand': total_lost,
            'infeasible_stations': infeasible_stations,
            'vehicles_after': vehicles_after
        }
    
    def _generate_benders_cut(self,
                               vehicle_counts: np.ndarray,
                               od_demand: np.ndarray,
                               relocations: Dict,
                               infeasible_stations: List[int]) -> Dict:
        """
        Generate a Benders cut based on infeasibility.
        
        The cut requires minimum vehicles at infeasible stations.
        """
        cut = {
            'stations': infeasible_stations,
            'min_vehicles': 0
        }
        
        # Minimum vehicles needed at each infeasible station
        for s in infeasible_stations:
            demand_at_s = od_demand[s, :].sum()
            cut['min_vehicles'] = max(cut['min_vehicles'], demand_at_s * 0.8)
        
        return cut


# ============================================================
# Comparison with MUF
# ============================================================

def compare_lbbd_vs_muf(vehicle_counts: np.ndarray, 
                        od_demand: np.ndarray,
                        verbose: bool = True):
    """
    Compare LBBD with MUF heuristic on total cost (relocation + lost demand).
    """
    print("\n" + "="*70)
    print("LBBD vs MUF COMPARISON")
    print("="*70)
    
    NS = len(vehicle_counts)
    config = LBBDConfig(
        num_stations=NS,
        vehicles_per_station=int(vehicle_counts.mean())
    )
    
    # LBBD
    print("\n--- LBBD ---")
    lbbd_solver = LBBDRelocationSolver(config)
    lbbd_result = lbbd_solver.solve(vehicle_counts, od_demand, verbose=verbose)
    
    # Calculate LBBD vehicles after relocation
    lbbd_vehicles_after = vehicle_counts.copy()
    for src, dst, count in lbbd_result.relocations:
        lbbd_vehicles_after[src] -= count
        lbbd_vehicles_after[dst] += count
    
    lbbd_lost = 0
    for i in range(NS):
        demand_from_i = od_demand[i, :].sum()
        if demand_from_i > lbbd_vehicles_after[i]:
            lbbd_lost += demand_from_i - lbbd_vehicles_after[i]
    
    lbbd_reloc_cost = sum(r[2] for r in lbbd_result.relocations) * config.relocation_cost
    lbbd_lost_cost = lbbd_lost * config.lost_demand_penalty
    lbbd_total = lbbd_reloc_cost + lbbd_lost_cost
    
    # MUF (simplified version)
    print("\n--- MUF Heuristic ---")
    muf_vehicles_after = vehicle_counts.copy()
    muf_relocations = []
    muf_reloc_cost = 0.0
    
    target = vehicle_counts.mean()
    
    # Find surplus and deficit stations
    surplus = {i: vehicle_counts[i] - target 
               for i in range(NS) if vehicle_counts[i] > target}
    deficit = {i: target - vehicle_counts[i] 
               for i in range(NS) if vehicle_counts[i] < target}
    
    # Sort by urgency
    surplus_list = sorted(surplus.items(), key=lambda x: -x[1])
    deficit_list = sorted(deficit.items(), key=lambda x: -x[1])
    
    for src, surplus_count in surplus_list:
        remaining_surplus = surplus_count
        for dst, deficit_count in deficit_list:
            if remaining_surplus <= 0:
                break
            remaining_deficit = deficit.get(dst, 0)
            if remaining_deficit <= 0:
                continue
            
            to_move = int(min(remaining_surplus, remaining_deficit, 50))
            if to_move > 0:
                muf_relocations.append((src, dst, to_move))
                muf_vehicles_after[src] -= to_move
                muf_vehicles_after[dst] += to_move
                muf_reloc_cost += to_move * config.relocation_cost
                remaining_surplus -= to_move
                deficit[dst] -= to_move
    
    # Calculate MUF lost demand
    muf_lost = 0
    for i in range(NS):
        demand_from_i = od_demand[i, :].sum()
        if demand_from_i > muf_vehicles_after[i]:
            muf_lost += demand_from_i - muf_vehicles_after[i]
    
    muf_lost_cost = muf_lost * config.lost_demand_penalty
    muf_total = muf_reloc_cost + muf_lost_cost
    
    print(f"Relocations: {len(muf_relocations)}")
    print(f"Vehicles relocated: {sum(r[2] for r in muf_relocations)}")
    print(f"Relocation cost: ${muf_reloc_cost:.2f}")
    print(f"Lost demand: {muf_lost:.0f}")
    print(f"Lost demand cost: ${muf_lost_cost:.2f}")
    print(f"Total cost: ${muf_total:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("FAIR COMPARISON (Relocation Cost + Lost Demand Cost)")
    print("="*70)
    print(f"{'Method':<10} {'Relocations':<12} {'Reloc Cost':<12} {'Lost Demand':<12} {'Lost Cost':<12} {'TOTAL':<12}")
    print("-"*70)
    
    lbbd_relocs = sum(r[2] for r in lbbd_result.relocations)
    muf_relocs = sum(r[2] for r in muf_relocations)
    
    print(f"{'LBBD':<10} {lbbd_relocs:<12} ${lbbd_reloc_cost:<11.0f} {lbbd_lost:<12.0f} ${lbbd_lost_cost:<11.0f} ${lbbd_total:<11.0f}")
    print(f"{'MUF':<10} {muf_relocs:<12} ${muf_reloc_cost:<11.0f} {muf_lost:<12.0f} ${muf_lost_cost:<11.0f} ${muf_total:<11.0f}")
    
    # Winner
    print("\n" + "-"*70)
    if lbbd_total < muf_total:
        diff = muf_total - lbbd_total
        print(f"✓ LBBD is better by ${diff:.0f} ({diff/muf_total*100:.1f}% savings)")
    else:
        diff = lbbd_total - muf_total
        print(f"✓ MUF is better by ${diff:.0f} ({diff/lbbd_total*100:.1f}% savings)")
    
    return lbbd_result, muf_relocations


# ============================================================
# Main Entry Point
# ============================================================

def run_lbbd_experiment(seed: int = 42, use_real_data: bool = True):
    """Run LBBD experiment with real NYC data or synthetic data."""
    np.random.seed(seed)
    
    print("="*70)
    print("LBBD RELOCATION EXPERIMENT")
    print("="*70)
    
    NS = 10
    
    if use_real_data:
        # Load real NYC Taxi OD data
        import os
        od_path = "data/processed/od_matrix_week.npy"
        
        if os.path.exists(od_path):
            od_week = np.load(od_path)  # (10, 10, 24, 7)
            print(f"\nLoaded real NYC OD data: {od_week.shape}")
            print(f"Total weekly demand: {od_week.sum():,.0f} trips")
            
            # Use peak hour demand (e.g., 8am on Monday)
            hour = 8
            day = 0
            od_demand = od_week[:, :, hour, day].astype(float)
            
            print(f"Using Hour {hour}, Day {day}")
            print(f"Hour demand: {od_demand.sum():,.0f} trips")
            
            # Initial vehicles: balanced 800 per station (统一配置)
            vehicles_per_station = 800
            vehicle_counts = np.full(NS, vehicles_per_station, dtype=float)
            
        else:
            print("Warning: NYC OD data not found, using synthetic data")
            use_real_data = False
    
    if not use_real_data:
        # Synthetic data (fallback)
        vehicle_counts = np.array([450, 400, 350, 350, 320, 200, 150, 100, 100, 80], dtype=float)
        od_demand = np.random.poisson(20, (NS, NS)).astype(float)
        for i in range(NS):
            od_demand[i, i] = 0
            if i >= 5:
                od_demand[i, :] *= 1.5
    
    print(f"\nInitial vehicle distribution: {vehicle_counts.astype(int)}")
    print(f"Total vehicles: {vehicle_counts.sum():.0f}")
    print(f"Total demand: {od_demand.sum():.0f}")
    
    # Run comparison
    lbbd_result, muf_result = compare_lbbd_vs_muf(vehicle_counts, od_demand)
    
    return lbbd_result, muf_result


if __name__ == "__main__":
    run_lbbd_experiment(seed=42, use_real_data=True)
