"""
ADP Comparison Experiment

Compares:
1. Myopic Policy: Greedy optimization without foresight
2. ADP-Guided Policy: Uses value function to evaluate states

The key insight is that ADP can:
- Prefer actions that lead to high-value future states
- Penalize actions that cause vehicle imbalance
- Consider time-of-day effects on future demand
"""

import numpy as np
from typing import Dict, Tuple
from adp_algorithm import (
    ADPState, ValueFunction, ADPTrainer, 
    extract_features, evaluate_policy
)


class ADPGuidedSimulator:
    """
    Simulator that uses ADP value function to guide relocation decisions.
    Uses real NYC Taxi OD data for demand.
    """
    
    def __init__(self, value_function: ValueFunction, num_stations: int = 10,
                 od_matrix_path: str = "data/processed/od_matrix_week.npy"):
        self.vf = value_function
        self.num_stations = num_stations
        self.vehicles_per_station = 300
        self.energy_per_vehicle = 21
        self.relocation_cost = 5.0
        
        # Load real NYC OD data
        import os
        if os.path.exists(od_matrix_path):
            self.od_week = np.load(od_matrix_path)
            self.use_real_data = True
            print(f"Simulator loaded NYC OD: {self.od_week.shape}")
        else:
            self.od_week = None
            self.use_real_data = False
    
    def create_initial_state(self, imbalanced: bool = False, day: int = 0) -> ADPState:
        """Create initial state."""
        if imbalanced:
            vehicle_counts = np.array([450, 400, 350, 350, 320, 200, 150, 100, 100, 80])
        else:
            vehicle_counts = np.full(self.num_stations, self.vehicles_per_station)
        
        energy_levels = vehicle_counts * self.energy_per_vehicle
        return ADPState(
            vehicle_counts=vehicle_counts.astype(float),
            energy_levels=energy_levels.astype(float),
            hour=0,
            day=day
        )
    
    def simulate_demand(self, state: ADPState) -> np.ndarray:
        """
        Generate OD demand using real NYC data patterns.
        """
        NS = self.num_stations
        
        if self.use_real_data:
            day_idx = state.day % 7
            od_demand = self.od_week[:, :, state.hour, day_idx].copy()
            # Scale to match fleet size
            scale_factor = (self.vehicles_per_station * NS) / (self.od_week[:, :, :, day_idx].sum() / 24 * 0.8)
            od_demand = np.random.poisson(od_demand * scale_factor)
        else:
            # Fallback synthetic demand
            hour = state.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_demand = 300
            elif 0 <= hour <= 5:
                base_demand = 60
            else:
                base_demand = 160
            od_demand = np.random.poisson(base_demand / (NS * NS), (NS, NS))
        
        return od_demand
    
    def apply_demand(self, state: ADPState, od_demand: np.ndarray) -> Tuple[ADPState, float, int]:
        """
        Apply demand and calculate reward.
        
        Returns:
            next_state, reward, lost_demand
        """
        NS = self.num_stations
        next_state = state.copy()
        
        total_revenue = 0.0
        total_trips = 0
        lost_demand = 0
        
        for i in range(NS):
            for j in range(NS):
                demand = int(od_demand[i, j])
                available = int(next_state.vehicle_counts[i])
                served = min(demand, available)
                
                next_state.vehicle_counts[i] -= served
                next_state.vehicle_counts[j] += served
                
                total_trips += served
                lost_demand += (demand - served)
                total_revenue += served * 12.0  # $12 per trip
        
        # Charging cost
        charging_cost = total_trips * 0.5 * 0.15
        
        # Lost demand penalty
        lost_penalty = lost_demand * 5.0
        
        reward = total_revenue - charging_cost - lost_penalty
        
        return next_state, reward, lost_demand
    
    def get_best_relocation(self, state: ADPState, max_relocations: int = 50) -> Tuple[int, int, int]:
        """
        Use value function to find best relocation (source, dest, count).
        
        Returns:
            (source, destination, count) or (-1, -1, 0) if no relocation improves value
        """
        NS = self.num_stations
        best_improvement = 0.0
        best_action = (-1, -1, 0)
        
        current_value = self.vf.predict(state)
        
        # Try relocating vehicles between each pair of stations
        for src in range(NS):
            available = int(state.vehicle_counts[src])
            if available < 10:
                continue
            
            for dst in range(NS):
                if src == dst:
                    continue
                
                # Try relocating 10 vehicles
                count = min(10, available - 5)  # Keep at least 5 at source
                
                # Create hypothetical next state
                test_state = state.copy()
                test_state.vehicle_counts[src] -= count
                test_state.vehicle_counts[dst] += count
                
                # Evaluate
                new_value = self.vf.predict(test_state)
                cost = count * self.relocation_cost
                
                improvement = new_value - current_value - cost
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = (src, dst, count)
        
        return best_action
    
    def simulate_day(self, use_adp: bool = False, imbalanced: bool = True) -> Dict:
        """
        Simulate one day of operation.
        
        Args:
            use_adp: If True, use ADP to guide relocation decisions
            imbalanced: If True, start with imbalanced vehicle distribution
        
        Returns:
            Daily metrics
        """
        state = self.create_initial_state(imbalanced=imbalanced)
        
        total_reward = 0.0
        total_trips = 0
        total_lost = 0
        total_relocations = 0
        relocation_cost = 0.0
        
        for hour in range(24):
            state.hour = hour
            
            # ADP-guided relocation (before demand arrives)
            if use_adp:
                for _ in range(5):  # Allow up to 5 relocation decisions per hour
                    src, dst, count = self.get_best_relocation(state)
                    if count > 0:
                        state.vehicle_counts[src] -= count
                        state.vehicle_counts[dst] += count
                        total_relocations += count
                        relocation_cost += count * self.relocation_cost
                    else:
                        break
            
            # Generate and apply demand
            demand = self.simulate_demand(state)
            state, reward, lost = self.apply_demand(state, demand)
            
            total_reward += reward
            total_lost += lost
        
        # Subtract relocation cost
        net_reward = total_reward - relocation_cost
        
        return {
            'net_reward': net_reward,
            'gross_reward': total_reward,
            'lost_demand': total_lost,
            'relocations': total_relocations,
            'relocation_cost': relocation_cost,
            'final_vehicle_std': state.vehicle_counts.std()
        }


def run_comparison_experiment(num_days: int = 30, seed: int = 42):
    """
    Run comparison between myopic and ADP-guided policies.
    """
    np.random.seed(seed)
    
    print("=" * 70)
    print("ADP vs MYOPIC POLICY COMPARISON")
    print("=" * 70)
    
    # First, train the value function
    print("\n--- Phase 1: Training Value Function ---")
    vf = ValueFunction(num_features=25)
    trainer = ADPTrainer(vf, num_stations=10)
    stats = trainer.train(num_episodes=500, verbose=True)
    
    # Create simulator
    sim = ADPGuidedSimulator(vf, num_stations=10)
    
    # Simulate with myopic policy
    print(f"\n--- Phase 2: Simulating {num_days} Days with Myopic Policy ---")
    myopic_results = []
    for day in range(num_days):
        result = sim.simulate_day(use_adp=False)
        myopic_results.append(result)
        if (day + 1) % 10 == 0:
            avg_reward = np.mean([r['net_reward'] for r in myopic_results[-10:]])
            print(f"  Day {day+1}: Avg Reward = ${avg_reward:,.0f}")
    
    # Simulate with ADP-guided policy
    np.random.seed(seed)  # Reset seed for fair comparison
    print(f"\n--- Phase 3: Simulating {num_days} Days with ADP-Guided Policy ---")
    adp_results = []
    for day in range(num_days):
        result = sim.simulate_day(use_adp=True)
        adp_results.append(result)
        if (day + 1) % 10 == 0:
            avg_reward = np.mean([r['net_reward'] for r in adp_results[-10:]])
            print(f"  Day {day+1}: Avg Reward = ${avg_reward:,.0f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    myopic_total = sum(r['net_reward'] for r in myopic_results)
    adp_total = sum(r['net_reward'] for r in adp_results)
    
    myopic_lost = sum(r['lost_demand'] for r in myopic_results)
    adp_lost = sum(r['lost_demand'] for r in adp_results)
    
    adp_relocations = sum(r['relocations'] for r in adp_results)
    adp_reloc_cost = sum(r['relocation_cost'] for r in adp_results)
    
    print(f"\n{'Metric':<25} {'Myopic':>15} {'ADP-Guided':>15} {'Difference':>15}")
    print("-" * 70)
    print(f"{'Total Profit':<25} ${myopic_total:>14,.0f} ${adp_total:>14,.0f} ${adp_total-myopic_total:>+14,.0f}")
    print(f"{'Lost Demand':<25} {myopic_lost:>15,} {adp_lost:>15,} {adp_lost-myopic_lost:>+15,}")
    print(f"{'Total Relocations':<25} {0:>15,} {adp_relocations:>15,} {adp_relocations:>+15,}")
    print(f"{'Relocation Cost':<25} ${0:>14,.0f} ${adp_reloc_cost:>14,.0f} ${adp_reloc_cost:>+14,.0f}")
    
    improvement = (adp_total - myopic_total) / myopic_total * 100
    print(f"\n** Improvement: {improvement:+.2f}% **")
    
    # Daily averages
    print(f"\nDaily Averages:")
    print(f"  Myopic: ${myopic_total/num_days:,.2f}/day")
    print(f"  ADP:    ${adp_total/num_days:,.2f}/day")
    
    return {
        'myopic': myopic_results,
        'adp': adp_results,
        'improvement_pct': improvement
    }


if __name__ == "__main__":
    results = run_comparison_experiment(num_days=30, seed=42)
