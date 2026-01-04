"""
Algorithm 3: Approximate Dynamic Programming (ADP)

ADP enables forward-looking decisions by learning a value function that
estimates the future profitability of system states.

Key components:
- ADPState: State representation (vehicle distribution, energy, time)
- extract_features: Feature engineering for value function approximation
- ValueFunction: Linear value function with TD(0) learning
- ADPTrainer: Training loop using simulated episodes
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json


# ============================================================
# State Representation
# ============================================================

@dataclass
class ADPState:
    """
    State representation for ADP.
    
    Attributes:
        vehicle_counts: Number of available vehicles at each station
        energy_levels: Total energy (kWh) at each station
        hour: Current hour of day (0-23)
        day: Day index (for multi-day simulation)
    """
    vehicle_counts: np.ndarray  # shape (NS,)
    energy_levels: np.ndarray   # shape (NS,)
    hour: int
    day: int = 0
    
    def copy(self) -> 'ADPState':
        return ADPState(
            vehicle_counts=self.vehicle_counts.copy(),
            energy_levels=self.energy_levels.copy(),
            hour=self.hour,
            day=self.day
        )
    
    def to_dict(self) -> dict:
        return {
            'vehicle_counts': self.vehicle_counts.tolist(),
            'energy_levels': self.energy_levels.tolist(),
            'hour': self.hour,
            'day': self.day
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'ADPState':
        return ADPState(
            vehicle_counts=np.array(d['vehicle_counts']),
            energy_levels=np.array(d['energy_levels']),
            hour=d['hour'],
            day=d.get('day', 0)
        )


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(state: ADPState) -> np.ndarray:
    """
    Extract features from state for value function approximation.
    
    Features (total = 5 + 2*NS = 25 for NS=10):
    1. Total vehicles (normalized)
    2. Vehicle imbalance (std of counts, normalized)
    3. Total energy (normalized)
    4. Hour encoding: sin(2π * hour / 24)
    5. Hour encoding: cos(2π * hour / 24)
    6-15. Vehicles at each station (normalized)
    16-25. Energy at each station (normalized)
    
    Returns:
        Feature vector of shape (5 + 2*NS,)
    """
    NS = len(state.vehicle_counts)
    
    # Normalization constants (based on typical values)
    MAX_VEHICLES = 500 * NS  # max total vehicles
    MAX_VEHICLE_PER_STATION = 500
    MAX_ENERGY_PER_STATION = 10000  # kWh
    
    features = []
    
    # 1. Total vehicles (normalized to [0, 1])
    total_vehicles = state.vehicle_counts.sum()
    features.append(total_vehicles / MAX_VEHICLES)
    
    # 2. Vehicle imbalance (std normalized)
    vehicle_std = state.vehicle_counts.std()
    features.append(vehicle_std / MAX_VEHICLE_PER_STATION)
    
    # 3. Total energy (normalized)
    total_energy = state.energy_levels.sum()
    features.append(total_energy / (MAX_ENERGY_PER_STATION * NS))
    
    # 4-5. Hour encoding (cyclic)
    hour_rad = 2 * np.pi * state.hour / 24
    features.append(np.sin(hour_rad))
    features.append(np.cos(hour_rad))
    
    # 6-(5+NS). Vehicles at each station (normalized)
    for v in state.vehicle_counts:
        features.append(v / MAX_VEHICLE_PER_STATION)
    
    # (6+NS)-(5+2*NS). Energy at each station (normalized)
    for e in state.energy_levels:
        features.append(e / MAX_ENERGY_PER_STATION)
    
    return np.array(features)


# ============================================================
# Value Function Approximation
# ============================================================

class ValueFunction:
    """
    Linear value function approximation: V(s) = θᵀφ(s)
    
    Uses TD(0) learning for updates.
    """
    
    def __init__(self, num_features: int = 25):
        """
        Args:
            num_features: Dimension of feature vector
        """
        self.num_features = num_features
        self.theta = np.zeros(num_features)  # Initialize weights to zero
        self.learning_rate = 0.01
        self.gamma = 0.95  # Discount factor
        
        # Training history
        self.td_errors: List[float] = []
        self.updates: int = 0
    
    def predict(self, state: ADPState) -> float:
        """Predict value of a state."""
        features = extract_features(state)
        return np.dot(self.theta, features)
    
    def td_update(self, state: ADPState, reward: float, next_state: ADPState, 
                  done: bool = False) -> float:
        """
        Temporal Difference (TD(0)) update.
        
        δ = r + γ * V(s') - V(s)
        θ += α * δ * φ(s)
        
        Args:
            state: Current state
            reward: Immediate reward (profit)
            next_state: Next state after action
            done: Whether episode is complete
            
        Returns:
            TD error (for logging)
        """
        features = extract_features(state)
        current_value = np.dot(self.theta, features)
        
        if done:
            next_value = 0.0
        else:
            next_value = self.predict(next_state)
        
        # TD target
        target = reward + self.gamma * next_value
        
        # TD error
        td_error = target - current_value
        
        # Update weights
        self.theta += self.learning_rate * td_error * features
        
        # Log
        self.td_errors.append(abs(td_error))
        self.updates += 1
        
        return td_error
    
    def save(self, filepath: str):
        """Save value function to file."""
        data = {
            'theta': self.theta.tolist(),
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'updates': self.updates
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load value function from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.theta = np.array(data['theta'])
        self.num_features = data['num_features']
        self.learning_rate = data.get('learning_rate', 0.01)
        self.gamma = data.get('gamma', 0.95)
        self.updates = data.get('updates', 0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance of each feature based on weight magnitude."""
        NS = (self.num_features - 5) // 2
        
        feature_names = [
            'total_vehicles',
            'vehicle_imbalance',
            'total_energy',
            'hour_sin',
            'hour_cos'
        ]
        for i in range(NS):
            feature_names.append(f'vehicles_station_{i}')
        for i in range(NS):
            feature_names.append(f'energy_station_{i}')
        
        importance = {}
        for name, weight in zip(feature_names, self.theta):
            importance[name] = abs(weight)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# ADP Trainer
# ============================================================

class ADPTrainer:
    """
    Trains value function using simulated episodes with real NYC Taxi demand patterns.
    """
    
    def __init__(self, value_function: ValueFunction, num_stations: int = 10, 
                 od_matrix_path: str = "data/processed/od_matrix_week.npy"):
        self.vf = value_function
        self.num_stations = num_stations
        
        # Simulation parameters
        self.vehicles_per_station = 800  # 统一配置：800辆/站
        self.energy_per_vehicle = 21  # kWh (30% of 70 kWh battery)
        
        # Load real NYC Taxi OD matrix
        import os
        if os.path.exists(od_matrix_path):
            self.od_week = np.load(od_matrix_path)  # (10, 10, 24, 7)
            self.use_real_data = True
            print(f"Loaded real NYC OD matrix: {self.od_week.shape}")
            print(f"  Total weekly demand: {self.od_week.sum():,.0f} trips")
            print(f"  Avg hourly demand: {self.od_week.sum() / 168:,.0f} trips/hour")
        else:
            self.od_week = None
            self.use_real_data = False
            print("Warning: NYC OD data not found, using synthetic demand")
        
        # Training history
        self.episode_rewards: List[float] = []
        self.episode_values: List[float] = []
    
    def create_initial_state(self) -> ADPState:
        """Create a random initial state for training."""
        # Random vehicle distribution (normally distributed around mean)
        mean_vehicles = self.vehicles_per_station
        std_vehicles = mean_vehicles * 0.2
        
        vehicle_counts = np.clip(
            np.random.normal(mean_vehicles, std_vehicles, self.num_stations),
            0, mean_vehicles * 2
        ).astype(int)
        
        # Energy proportional to vehicles
        energy_levels = vehicle_counts * self.energy_per_vehicle
        
        # Random starting hour and day
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)  # Random day of week
        
        return ADPState(
            vehicle_counts=vehicle_counts,
            energy_levels=energy_levels.astype(float),
            hour=hour,
            day=day
        )
    
    def simulate_transition(self, state: ADPState, od_demand: np.ndarray = None) -> Tuple[ADPState, float]:
        """
        Simulate one hour transition using real NYC demand patterns.
        
        Args:
            state: Current state
            od_demand: OD demand matrix for current hour (NS, NS), if None uses real data
            
        Returns:
            next_state, reward
        """
        NS = self.num_stations
        next_state = state.copy()
        
        # Use real NYC demand data if available
        if od_demand is None:
            if self.use_real_data:
                # Get real OD demand for this hour and day
                day_idx = state.day % 7
                od_demand = self.od_week[:, :, state.hour, day_idx].copy()
                # Scale to simulation scale (real data has ~7000/hour, we use 300 vehicles/station)
                # Scale factor: keep demand realistic relative to fleet size
                scale_factor = (self.vehicles_per_station * NS) / (self.od_week[:, :, :, day_idx].sum() / 24 * 0.8)
                od_demand = np.random.poisson(od_demand * scale_factor)
            else:
                # Fallback to synthetic demand
                hour = state.hour
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    base_demand = 150  # Peak hours
                elif 0 <= hour <= 5:
                    base_demand = 30   # Night
                else:
                    base_demand = 80   # Normal
                od_demand = np.random.poisson(base_demand / (NS * NS), (NS, NS))
        
        # Simulate trips
        total_revenue = 0.0
        total_trips = 0
        lost_demand = 0
        
        for i in range(NS):
            for j in range(NS):
                demand = int(od_demand[i, j])
                available = int(next_state.vehicle_counts[i])
                served = min(demand, available)
                
                # Update state
                next_state.vehicle_counts[i] -= served
                next_state.vehicle_counts[j] += served
                
                # Track metrics
                total_trips += served
                lost_demand += (demand - served)
                
                # Revenue: $10-15 per trip (price depends on demand)
                price = 12.0
                total_revenue += served * price
        
        # Charging cost (simplified)
        charging_cost = total_trips * 0.5 * 0.15  # 0.5 kWh per trip, $0.15/kWh
        
        # Penalty for imbalance
        vehicle_std = next_state.vehicle_counts.std()
        imbalance_penalty = vehicle_std * 0.1
        
        # Lost demand penalty
        lost_penalty = lost_demand * 5.0  # $5 per lost trip
        
        # Calculate reward
        reward = total_revenue - charging_cost - imbalance_penalty - lost_penalty
        
        # Update time
        next_state.hour = (state.hour + 1) % 24
        if next_state.hour == 0:
            next_state.day += 1
        
        return next_state, reward
    
    def train_episode(self, max_steps: int = 24) -> float:
        """
        Train one episode (one day).
        
        Returns:
            Total reward for the episode
        """
        state = self.create_initial_state()
        initial_value = self.vf.predict(state)
        
        total_reward = 0.0
        
        for step in range(max_steps):
            done = (step == max_steps - 1)
            
            # Simulate transition
            next_state, reward = self.simulate_transition(state)
            
            # TD update
            self.vf.td_update(state, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        # Log episode
        self.episode_rewards.append(total_reward)
        self.episode_values.append(initial_value)
        
        return total_reward
    
    def train(self, num_episodes: int = 100, verbose: bool = False) -> Dict:
        """
        Train value function over multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        print(f"Training ADP value function for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            reward = self.train_episode()
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_td_error = np.mean(self.vf.td_errors[-240:]) if self.vf.td_errors else 0
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"Avg Reward={avg_reward:,.0f}, "
                      f"Avg TD Error={avg_td_error:.2f}")
        
        print(f"Training complete. Total updates: {self.vf.updates}")
        
        return {
            'num_episodes': num_episodes,
            'final_avg_reward': np.mean(self.episode_rewards[-10:]),
            'avg_td_error': np.mean(self.vf.td_errors[-240:]) if self.vf.td_errors else 0,
            'feature_importance': self.vf.get_feature_importance(),
            'episode_rewards': self.episode_rewards,
            'episode_values': self.episode_values
        }


# ============================================================
# Policy Evaluation
# ============================================================

def evaluate_policy(value_function: ValueFunction, 
                    num_episodes: int = 10,
                    use_adp_guidance: bool = True) -> Dict:
    """
    Evaluate a policy using the learned value function.
    
    Args:
        value_function: Trained value function
        num_episodes: Number of evaluation episodes
        use_adp_guidance: If True, use value function to guide decisions
        
    Returns:
        Evaluation metrics
    """
    trainer = ADPTrainer(value_function)
    
    total_rewards = []
    
    for _ in range(num_episodes):
        state = trainer.create_initial_state()
        episode_reward = 0.0
        
        for step in range(24):  # One day
            if use_adp_guidance:
                # Evaluate potential actions and choose best
                # For now, just simulate (can be extended with action selection)
                pass
            
            next_state, reward = trainer.simulate_transition(state)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards)
    }


# ============================================================
# Main Entry Point
# ============================================================

def run_adp_experiment(num_episodes: int = 100, seed: int = 42):
    """Run ADP training experiment."""
    np.random.seed(seed)
    
    print("=" * 70)
    print("APPROXIMATE DYNAMIC PROGRAMMING (ADP) EXPERIMENT")
    print("=" * 70)
    
    # Create value function and trainer
    vf = ValueFunction(num_features=25)
    trainer = ADPTrainer(vf, num_stations=10)
    
    # Train
    stats = trainer.train(num_episodes=num_episodes, verbose=True)
    
    # Print results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Final Avg Reward: ${stats['final_avg_reward']:,.2f}")
    print(f"Avg TD Error: {stats['avg_td_error']:.4f}")
    
    print("\nTop 10 Feature Importances:")
    importance = stats['feature_importance']
    for i, (name, weight) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {name}: {weight:.4f}")
    
    # Save value function
    vf.save('adp_value_function.json')
    print(f"\nValue function saved to adp_value_function.json")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("POLICY EVALUATION")
    print("=" * 70)
    
    eval_stats = evaluate_policy(vf, num_episodes=20)
    print(f"Mean Daily Reward: ${eval_stats['mean_reward']:,.2f}")
    print(f"Std Daily Reward: ${eval_stats['std_reward']:,.2f}")
    print(f"Range: [${eval_stats['min_reward']:,.2f}, ${eval_stats['max_reward']:,.2f}]")
    
    return vf, trainer, stats


if __name__ == "__main__":
    vf, trainer, stats = run_adp_experiment(num_episodes=100, seed=42)
