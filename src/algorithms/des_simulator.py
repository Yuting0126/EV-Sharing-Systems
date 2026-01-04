"""
Algorithm 4: Discrete-Event Simulation (DES) for EV Sharing System
====================================================================

This module implements a discrete-event simulation framework to evaluate
the performance of the EV sharing optimization algorithms under stochastic
user demand.

Key Features:
- Event-driven architecture with priority queue
- Poisson-based user arrival process (calibrated from NYC Taxi data)
- Integration with Algorithm 1 (MIQP) and Algorithm 2 (MUF)
- Comprehensive performance metrics
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
import pandas as pd


# ============================================================
# Enums and Constants
# ============================================================

class EventType(Enum):
    """Types of events in the simulation"""
    USER_ARRIVAL = auto()       # A user wants to rent a vehicle
    TRIP_END = auto()           # A trip completes, vehicle arrives at destination
    CHARGING_COMPLETE = auto()  # Vehicle finishes charging
    RELOCATION_END = auto()     # Relocation vehicle arrives at destination
    DECISION_EPOCH = auto()     # Hourly decision point for pricing/charging


class VehicleStatus(Enum):
    """Status of a vehicle"""
    IDLE = auto()           # Available for rent
    CHARGING = auto()       # Currently charging
    RENTED = auto()         # In use by customer
    RELOCATING = auto()     # Being relocated by operator


# ============================================================
# Data Classes
# ============================================================

@dataclass(order=True)
class Event:
    """
    Represents a single event in the simulation.
    Events are ordered by time for the priority queue.
    """
    time: float
    event_type: EventType = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)
    
    def __repr__(self):
        return f"Event({self.time:.2f}, {self.event_type.name}, {self.data})"


@dataclass
class Vehicle:
    """Represents a single EV in the system"""
    vehicle_id: int
    soc: float = 1.0  # State of charge (0.0 to 1.0)
    status: VehicleStatus = VehicleStatus.IDLE
    
    @property
    def is_available(self) -> bool:
        """Check if vehicle is available for rent (must be IDLE and have enough charge)"""
        return self.status == VehicleStatus.IDLE and self.soc >= 0.2  # Min 20% charge


@dataclass
class Trip:
    """Represents a trip in progress"""
    vehicle: Vehicle
    origin: int
    destination: int
    start_time: float
    end_time: float
    price: float


# ============================================================
# Event Queue
# ============================================================

class EventQueue:
    """Priority queue for events, ordered by time"""
    
    def __init__(self):
        self._queue: List[Event] = []
    
    def push(self, event: Event):
        """Add an event to the queue"""
        heapq.heappush(self._queue, event)
    
    def pop(self) -> Event:
        """Remove and return the earliest event"""
        return heapq.heappop(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0
    
    def peek(self) -> Optional[Event]:
        """View the next event without removing it"""
        return self._queue[0] if self._queue else None
    
    def __len__(self) -> int:
        return len(self._queue)


# ============================================================
# System State
# ============================================================

class SystemState:
    """
    Maintains the complete state of the EV sharing system.
    """
    
    def __init__(self, num_stations: int, vehicles_per_station: int = 100):
        self.num_stations = num_stations
        self.current_time = 0.0
        
        # Vehicles at each station (list of Vehicle objects)
        self.vehicles_at_station: Dict[int, List[Vehicle]] = {
            i: [Vehicle(vehicle_id=i*1000 + j) 
                for j in range(vehicles_per_station)]
            for i in range(num_stations)
        }
        
        # Trips in progress
        self.trips_in_progress: List[Trip] = []
        
        # Vehicles being relocated
        self.relocations_in_progress: List[dict] = []
        
        # Metrics
        self.total_revenue = 0.0
        self.total_trips = 0
        self.lost_demand = 0
        self.total_charging_cost = 0.0
        self.total_relocation_cost = 0.0
        
        # Current pricing (set by MIQP, default $10)
        self.current_prices: np.ndarray = np.full((num_stations, num_stations), 10.0)
        
        # Charging rate (kW per charger)
        self.charging_power = 7.0  # kW
        self.battery_capacity = 21.0  # kWh (from config: E_max=21)
    
    def get_available_vehicle(self, station: int) -> Optional[Vehicle]:
        """Get an available vehicle at a station, or None if none available"""
        for vehicle in self.vehicles_at_station[station]:
            if vehicle.is_available:
                return vehicle
        return None
    
    def count_vehicles(self, station: int) -> int:
        """Count total vehicles at a station"""
        return len(self.vehicles_at_station[station])
    
    def count_available(self, station: int) -> int:
        """Count available vehicles at a station"""
        return sum(1 for v in self.vehicles_at_station[station] if v.is_available)
    
    def get_total_energy(self, station: int) -> float:
        """Get total energy (kWh) stored at a station"""
        return sum(v.soc * self.battery_capacity for v in self.vehicles_at_station[station])
    
    def get_summary(self) -> dict:
        """Get summary statistics of current state"""
        return {
            'time': self.current_time,
            'total_vehicles': sum(len(v) for v in self.vehicles_at_station.values()),
            'vehicles_per_station': [len(self.vehicles_at_station[i]) for i in range(self.num_stations)],
            'available_per_station': [self.count_available(i) for i in range(self.num_stations)],
            'trips_in_progress': len(self.trips_in_progress),
            'total_revenue': self.total_revenue,
            'total_trips': self.total_trips,
            'lost_demand': self.lost_demand,
            'service_rate': self.total_trips / (self.total_trips + self.lost_demand) 
                           if (self.total_trips + self.lost_demand) > 0 else 1.0
        }


# ============================================================
# Demand Generator
# ============================================================

class DemandGenerator:
    """
    Generates user arrivals based on OD matrix and price sensitivity.
    """
    def __init__(self, od_matrix: np.ndarray, b: float = 2.0):
        self.od_matrix = od_matrix.astype(float) # This is 'a' in a - b*p
        self.num_stations = od_matrix.shape[0]
        self.b = b
        self.num_days = od_matrix.shape[3] if len(od_matrix.shape) == 4 else 1
    
    def get_arrival_rate(self, current_time: float, current_prices: np.ndarray) -> np.ndarray:
        day = int(current_time / (24 * 60)) % self.num_days
        hour = int(current_time / 60) % 24
        
        if len(self.od_matrix.shape) == 4:
            a = self.od_matrix[:, :, hour, day]
        else:
            a = self.od_matrix[:, :, hour]
            
        # q = a - b * p
        rate = (a - self.b * current_prices) / 60.0
        return np.maximum(0.0001, rate) # Small floor to keep simulation moving

    def get_next_arrival(self, current_time: float, rng: np.random.Generator, current_prices: np.ndarray) -> Optional[Event]:
        rate_matrix = self.get_arrival_rate(current_time, current_prices)
        total_rate = rate_matrix.sum()
        
        if total_rate <= 0:
            return None
            
        inter_arrival = rng.exponential(1.0 / total_rate)
        arrival_time = current_time + inter_arrival
        
        # Sample OD pair
        flat_idx = rng.choice(self.num_stations**2, p=rate_matrix.flatten() / total_rate)
        o, d = divmod(flat_idx, self.num_stations)
        
        return Event(
            time=arrival_time, 
            event_type=EventType.USER_ARRIVAL, 
            data={'origin': o, 'destination': d}
        )


# ============================================================
# DES Simulator
# ============================================================

class EVSharingSimulator:
    """
    Main discrete-event simulation engine for EV sharing system.
    """
    
    def __init__(self, 
                 num_stations: int = 10,
                 vehicles_per_station: int = 100,
                 od_matrix: Optional[np.ndarray] = None,
                 energy_matrix: Optional[np.ndarray] = None,
                 enable_relocation: bool = True,
                 external_prices: Optional[np.ndarray] = None,
                 b_elasticity: float = 2.0,
                 seed: int = 42):
        """
        Args:
            num_stations: Number of stations in the system
            vehicles_per_station: Initial vehicles per station
            od_matrix: OD demand matrix (NS, NS, 24)
            energy_consumption: Average energy per trip (kWh)
            enable_relocation: Whether to enable MUF relocation
            external_prices: (NS, NS, 24) Optional pricing matrix from MIQP
            seed: Random seed for reproducibility
        """
        self.state = SystemState(num_stations, vehicles_per_station)
        self.event_queue = EventQueue()
        self.rng = np.random.default_rng(seed)
        self.energy_matrix = energy_matrix
        self.enable_relocation = enable_relocation
        self.external_prices = external_prices
        self.num_stations = num_stations
        
        # Demand generator
        if od_matrix is not None:
            self.demand_gen = DemandGenerator(od_matrix, b=b_elasticity)
        else:
            # Default: uniform demand
            self.demand_gen = DemandGenerator(
                np.ones((num_stations, num_stations, 24)) * 10, b=b_elasticity
            )
        
        # Trip duration (minutes) - mean 20 min
        self.trip_duration_mean = 20.0
        
        # Build distance matrix between stations (km)
        # Based on NYC Manhattan grid layout, stations are roughly 2-15km apart
        self.distance_matrix = self._build_distance_matrix(num_stations)
        
        # Average speed for relocation (km/h) - city driving with traffic
        self.relocation_speed_kmh = 25.0
        
        # Charging cost ($/kWh)
        self.charging_cost_rate = 0.15
        
        # Relocation cost ($/trip)
        self.relocation_cost_per_trip = 5.0
        
        # History for analysis
        self.event_log: List[dict] = []
    
    def _build_distance_matrix(self, num_stations: int) -> np.ndarray:
        """
        Build a distance matrix between stations based on NYC geography.
        Approximates Manhattan as a 5x2 grid of zones.
        """
        # NYC Manhattan approximate zone coordinates (simplified grid)
        # Stations 0-9 arranged in a 5x2 grid spanning ~10km x 4km
        zone_coords = np.array([
            [0, 0], [2, 0], [4, 0], [6, 0], [8, 0],   # Lower row
            [0, 4], [2, 4], [4, 4], [6, 4], [8, 4],   # Upper row
        ])[:num_stations]
        
        # Calculate Euclidean distances (km)
        dist_matrix = np.zeros((num_stations, num_stations))
        for i in range(num_stations):
            for j in range(num_stations):
                if i != j:
                    dist = np.sqrt(np.sum((zone_coords[i] - zone_coords[j])**2))
                    # Add 30% for non-direct routes in city
                    dist_matrix[i, j] = dist * 1.3
        
        return dist_matrix
    
    def get_travel_time(self, origin: int, destination: int) -> float:
        """
        Calculate travel time between two stations in minutes.
        """
        if origin == destination:
            return 0.0
        
        distance_km = self.distance_matrix[origin, destination]
        travel_time_hours = distance_km / self.relocation_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        # Minimum 5 minutes, maximum 30 minutes
        return max(5.0, min(30.0, travel_time_minutes))
    
    def initialize(self):
        """Initialize simulation by scheduling initial events"""
        # Schedule first user arrival
        first_arrival = self.demand_gen.get_next_arrival(0.0, self.rng, self.state.current_prices)
        if first_arrival:
            self.event_queue.push(first_arrival)
        
        # Schedule decision epochs (every hour)
        for hour in range(24):
            self.event_queue.push(Event(
                time=hour * 60.0,
                event_type=EventType.DECISION_EPOCH,
                data={'hour': hour}
            ))
    
    def handle_user_arrival(self, event: Event):
        """Handle a user arrival event"""
        origin = event.data['origin']
        destination = event.data['destination']
        
        # Try to find an available vehicle
        vehicle = self.state.get_available_vehicle(origin)
        if vehicle:
            # Serve
            vehicle.status = VehicleStatus.RENTED
            self.state.vehicles_at_station[origin].remove(vehicle)
            
            price = self.state.current_prices[origin, destination]
            trip_duration = self.rng.exponential(self.trip_duration_mean)
            end_time = event.time + trip_duration
            
            # Create trip
            trip = Trip(
                vehicle=vehicle,
                origin=origin,
                destination=destination,
                start_time=event.time,
                end_time=end_time,
                price=price
            )
            self.state.trips_in_progress.append(trip)
            
            # Schedule trip end
            self.event_queue.push(Event(
                time=end_time,
                event_type=EventType.TRIP_END,
                data={'trip': trip}
            ))
            
            self.log_event('TRIP_START', event.time, {
                'origin': origin, 'destination': destination, 
                'price': price, 'vehicle_id': vehicle.vehicle_id
            })
        else:
            self.state.lost_demand += 1
            self.log_event('LOST_DEMAND', event.time, 
                          {'origin': origin, 'destination': destination})
        
        # Schedule next arrival
        next_evt = self.demand_gen.get_next_arrival(event.time, self.rng, self.state.current_prices)
        if next_evt and next_evt.time < self.end_time:
            self.event_queue.push(next_evt)
    
    def handle_trip_end(self, event: Event):
        """Handle a trip completion event"""
        trip = event.data['trip']
        vehicle = trip.vehicle
        
        # Update vehicle state
        vehicle.status = VehicleStatus.IDLE
        
        # Energy consumption from matrix
        if self.energy_matrix is not None:
            hour = int(event.time / 60) % 24
            energy_consumed = self.energy_matrix[trip.origin, trip.destination, hour]
        else:
            energy_consumed = 0.5 # Default energy consumption if no matrix provided
            
        vehicle.soc = max(0.0, vehicle.soc - energy_consumed / self.state.battery_capacity)
        
        # Add vehicle to destination station
        self.state.vehicles_at_station[trip.destination].append(vehicle)
        
        # Remove from trips in progress
        if trip in self.state.trips_in_progress:
            self.state.trips_in_progress.remove(trip)
        
        # Update metrics
        self.state.total_revenue += trip.price
        self.state.total_trips += 1
        
        self.log_event('TRIP_END', event.time, {
            'origin': trip.origin, 'destination': trip.destination,
            'price': trip.price, 'vehicle_soc': vehicle.soc
        })
        
        # If vehicle needs charging, schedule it
        if vehicle.soc < 0.8:  # Charge if below 80%
            self.schedule_charging(vehicle, trip.destination, event.time)
    
    def schedule_charging(self, vehicle: Vehicle, station: int, current_time: float):
        """Schedule vehicle charging"""
        if vehicle.soc >= 1.0:
            return
        
        vehicle.status = VehicleStatus.CHARGING
        
        # Calculate charging time
        energy_needed = (1.0 - vehicle.soc) * self.state.battery_capacity
        charging_time = energy_needed / self.state.charging_power * 60  # minutes
        
        self.event_queue.push(Event(
            time=current_time + charging_time,
            event_type=EventType.CHARGING_COMPLETE,
            data={'vehicle': vehicle, 'station': station, 'energy': energy_needed}
        ))
    
    def handle_charging_complete(self, event: Event):
        """Handle charging completion event"""
        vehicle = event.data['vehicle']
        energy = event.data['energy']
        
        vehicle.soc = 1.0
        vehicle.status = VehicleStatus.IDLE
        
        # Update charging cost
        self.state.total_charging_cost += energy * self.charging_cost_rate
        
        self.log_event('CHARGING_COMPLETE', event.time, {
            'vehicle_id': vehicle.vehicle_id, 'energy': energy
        })
    
    def handle_decision_epoch(self, event: Event):
        """
        Handle hourly decision point.
        This is where we call Algorithm 1 (MIQP) and Algorithm 2 (MUF).
        """
        hour = event.data['hour']
        
        # Log current state
        summary = self.state.get_summary()
        self.log_event('DECISION_EPOCH', event.time, summary)
        
        # === Dynamic Pricing ===
        if self.external_prices is not None:
            # Shift prices from MIQP results
            self.state.current_prices = self.external_prices[:, :, hour]
        else:
            # Fallback to simple surge pricing
            for i in range(self.state.num_stations):
                avail_i = self.state.count_available(i)
                for j in range(self.state.num_stations):
                    base_price = 10.0
                    if avail_i < 100:
                        self.state.current_prices[i, j] = base_price * 1.5
                    elif avail_i > 400:
                        self.state.current_prices[i, j] = base_price * 0.8
                    else:
                        self.state.current_prices[i, j] = base_price
        
        # === MUF Relocation (Algorithm 2) ===
        if self.enable_relocation:
            self.execute_muf_relocation(event.time)
    
    def execute_muf_relocation(self, current_time: float):
        """
        Execute MUF (Most Urgent First) relocation strategy.
        Move vehicles from surplus stations to deficit stations.
        """
        NS = self.state.num_stations
        target_per_station = 300  # Target vehicle count
        
        # Calculate surplus and deficit
        surplus = []  # (station, excess_count, available_vehicles)
        deficit = []  # (station, needed_count)
        
        for i in range(NS):
            count = self.state.count_vehicles(i)
            available = [v for v in self.state.vehicles_at_station[i] if v.is_available]
            
            if count > target_per_station + 50:
                excess = count - target_per_station
                surplus.append((i, excess, available[:excess]))
            elif count < target_per_station - 50:
                needed = target_per_station - count
                deficit.append((i, needed))
        
        # Sort deficit by urgency (most needed first)
        deficit.sort(key=lambda x: x[1], reverse=True)
        
        relocations_scheduled = 0
        max_relocations_per_epoch = 50  # Limit to avoid overwhelming
        
        for (dest_station, needed) in deficit:
            if relocations_scheduled >= max_relocations_per_epoch:
                break
            
            for (src_station, excess, vehicles) in surplus:
                if relocations_scheduled >= max_relocations_per_epoch:
                    break
                if not vehicles:
                    continue
                
                # Move vehicles
                to_move = min(needed, len(vehicles), 10)  # Max 10 per OD pair
                for _ in range(to_move):
                    if not vehicles:
                        break
                    
                    vehicle = vehicles.pop()
                    vehicle.status = VehicleStatus.RELOCATING
                    self.state.vehicles_at_station[src_station].remove(vehicle)
                    
                    # Schedule relocation arrival - 动态计算行程时间
                    travel_time = self.get_travel_time(src_station, dest_station)
                    arrival_time = current_time + travel_time
                    self.event_queue.push(Event(
                        time=arrival_time,
                        event_type=EventType.RELOCATION_END,
                        data={'vehicle': vehicle, 'destination': dest_station, 'origin': src_station, 'travel_time': travel_time}
                    ))
                    
                    relocations_scheduled += 1
                    needed -= 1
                
                if needed <= 0:
                    break
        
        if relocations_scheduled > 0:
            self.log_event('RELOCATION_BATCH', current_time, {
                'count': relocations_scheduled
            })

    
    def handle_relocation_end(self, event: Event):
        """Handle relocation completion event"""
        vehicle = event.data['vehicle']
        destination = event.data['destination']
        
        vehicle.status = VehicleStatus.IDLE
        self.state.vehicles_at_station[destination].append(vehicle)
        
        self.state.total_relocation_cost += self.relocation_cost_per_trip
        
        self.log_event('RELOCATION_END', event.time, {
            'vehicle_id': vehicle.vehicle_id, 'destination': destination
        })
    
    def log_event(self, event_type: str, time: float, data: dict):
        """Log an event for analysis"""
        self.event_log.append({
            'type': event_type,
            'time': time,
            'hour': int(time / 60) % 24,
            **data
        })
    
    def run(self, duration_minutes: float = 24 * 60):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration_minutes: Simulation duration in minutes (default: 24 hours)
        
        Returns:
            dict: Performance metrics
        """
        self.end_time = duration_minutes
        self.initialize()
        
        print(f"Starting DES simulation for {duration_minutes/60:.1f} hours...")
        
        event_count = 0
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            
            if event.time > self.end_time:
                break
            
            self.state.current_time = event.time
            
            # Dispatch to appropriate handler
            if event.event_type == EventType.USER_ARRIVAL:
                self.handle_user_arrival(event)
            elif event.event_type == EventType.TRIP_END:
                self.handle_trip_end(event)
            elif event.event_type == EventType.CHARGING_COMPLETE:
                self.handle_charging_complete(event)
            elif event.event_type == EventType.RELOCATION_END:
                self.handle_relocation_end(event)
            elif event.event_type == EventType.DECISION_EPOCH:
                self.handle_decision_epoch(event)
            
            event_count += 1
            
            # Progress update
            if event_count % 10000 == 0:
                print(f"  Processed {event_count} events, time={event.time/60:.1f}h")
        
        print(f"Simulation complete. Processed {event_count} events.")
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Get simulation results"""
        summary = self.state.get_summary()
        
        profit = (self.state.total_revenue 
                  - self.state.total_charging_cost 
                  - self.state.total_relocation_cost)
        
        return {
            'duration_hours': self.state.current_time / 60,
            'total_trips': self.state.total_trips,
            'lost_demand': self.state.lost_demand,
            'service_rate': summary['service_rate'],
            'total_revenue': self.state.total_revenue,
            'charging_cost': self.state.total_charging_cost,
            'relocation_cost': self.state.total_relocation_cost,
            'net_profit': profit,
            'avg_revenue_per_trip': self.state.total_revenue / max(1, self.state.total_trips),
            'final_state': summary
        }


# ============================================================
# Main Entry Point
# ============================================================

def run_des_experiment(num_days: int = 1, enable_relocation: bool = True, seed: int = 42):
    """
    Run DES experiment with real NYC Taxi data.
    """
    import os
    
    reloc_str = "WITH" if enable_relocation else "WITHOUT"
    print(f"\n{'='*70}")
    print(f"DES SIMULATION ({reloc_str} RELOCATION)")
    print(f"{'='*70}")
    
    # Load OD matrix
    od_file = "data/processed/od_matrix_week.npy"
    if os.path.exists(od_file):
        od_week = np.load(od_file)  # (10, 10, 24, 7)
        if num_days > 1:
            # Use multi-day data
            od_matrix = od_week[:, :, :, :min(num_days, 7)]
            print(f"Loaded {min(num_days, 7)}-day OD matrix: {od_matrix.shape}")
        else:
            od_matrix = od_week[:, :, :, 0]  # Use first day only
            print(f"Loaded OD matrix: {od_matrix.shape}")
    else:
        print("Warning: OD matrix not found, using synthetic data")
        od_matrix = np.ones((10, 10, 24)) * 50
    
    # Create simulator
    sim = EVSharingSimulator(
        num_stations=10,
        vehicles_per_station=800,  # 统一配置：800辆/站
        od_matrix=od_matrix,
        enable_relocation=enable_relocation,
        seed=seed
    )
    
    # Run simulation
    duration = num_days * 24 * 60  # minutes
    results = sim.run(duration_minutes=duration)
    
    # Print results
    print(f"\n--- Results ({reloc_str} Relocation) ---")
    print(f"Total Trips: {results['total_trips']:,}")
    print(f"Lost Demand: {results['lost_demand']:,}")
    print(f"Service Rate: {results['service_rate']*100:.1f}%")
    print(f"Total Revenue: ${results['total_revenue']:,.2f}")
    print(f"Charging Cost: ${results['charging_cost']:,.2f}")
    print(f"Relocation Cost: ${results['relocation_cost']:,.2f}")
    print(f"Net Profit: ${results['net_profit']:,.2f}")
    
    return results, sim


def run_comparison_experiment(num_days: int = 1, seed: int = 42):
    """
    Run comparison experiment: with relocation vs without relocation.
    """
    print("=" * 70)
    print("DES COMPARISON EXPERIMENT: With vs Without Relocation")
    print("=" * 70)
    
    # Run WITH relocation
    results_with, sim_with = run_des_experiment(num_days, enable_relocation=True, seed=seed)
    
    # Run WITHOUT relocation
    results_without, sim_without = run_des_experiment(num_days, enable_relocation=False, seed=seed)
    
    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'With Reloc':>15} {'Without Reloc':>15} {'Difference':>15}")
    print("-" * 70)
    
    metrics = [
        ('Total Trips', 'total_trips', '{:,}'),
        ('Lost Demand', 'lost_demand', '{:,}'),
        ('Service Rate', 'service_rate', '{:.1%}'),
        ('Total Revenue', 'total_revenue', '${:,.0f}'),
        ('Relocation Cost', 'relocation_cost', '${:,.0f}'),
        ('Net Profit', 'net_profit', '${:,.0f}'),
    ]
    
    for name, key, fmt in metrics:
        val_with = results_with[key]
        val_without = results_without[key]
        
        if key == 'service_rate':
            diff_str = f"{(val_with - val_without)*100:+.1f}%"
        elif 'cost' in key or 'revenue' in key or 'profit' in key:
            diff_str = f"${val_with - val_without:+,.0f}"
        else:
            diff_str = f"{val_with - val_without:+,}"
        
        print(f"{name:<25} {fmt.format(val_with):>15} {fmt.format(val_without):>15} {diff_str:>15}")
    
    # Save comparison results
    comparison = {
        'with_relocation': results_with,
        'without_relocation': results_without,
        'advantage': {
            'profit_diff': results_with['net_profit'] - results_without['net_profit'],
            'service_rate_diff': results_with['service_rate'] - results_without['service_rate'],
            'lost_demand_reduction': results_without['lost_demand'] - results_with['lost_demand']
        }
    }
    
    print(f"\n** Relocation Advantage: ${comparison['advantage']['profit_diff']:,.0f} **")
    print(f"** Service Rate Improvement: {comparison['advantage']['service_rate_diff']*100:.1f}% **")
    
    return comparison


if __name__ == "__main__":
    comparison = run_comparison_experiment(num_days=1, seed=42)
