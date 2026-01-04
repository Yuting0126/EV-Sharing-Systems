import numpy as np
from . import config
from .power_grid import PowerGrid

class EVSharingEnv:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.num_stations = config.NUM_STATIONS
        self.time_steps = config.TIME_PERIODS
        self.current_step = 0
        
        # 初始化电网
        self.grid = PowerGrid(data_loader)
        
        # 状态变量
        self.vehicles = np.zeros(self.num_stations, dtype=int) # N_i
        self.energy = np.zeros(self.num_stations)              # E_i (kWh)
        
        # 累计指标
        self.total_profit = 0.0
        self.lost_sales = 0
        self.total_demand = 0
        self.total_charging_cost = 0.0 # 新增：累计充电成本
        
        self.reset()

    def reset(self):
        self.current_step = 0
        initial_vehicles, initial_energy = self.loader.get_initial_state()
        self.vehicles = initial_vehicles.astype(int)
        self.energy = initial_energy # 初始能量
        
        self.total_profit = 0.0
        self.lost_sales = 0
        self.total_demand = 0
        self.total_charging_cost = 0.0
        
        return self._get_state()

    def _get_state(self):
        return {
            't': self.current_step,
            'vehicles': self.vehicles.copy(),
            'energy': self.energy.copy()
        }

    def step(self, actions):
        """
        执行一步仿真
        actions: 字典，包含:
            - 'pricing': (8, 8) 价格矩阵
            - 'relocation': (8, 8) 重定位数量矩阵 (i -> j)
            - 'charging': (8,) 充电量 (暂简化)
        """
        t = self.current_step
        pricing = actions.get('pricing', np.ones((8, 8)) * config.BASE_PRICE)
        relocation = actions.get('relocation', np.zeros((8, 8), dtype=int))
        
        # 1. 处理重定位 (Relocation)
        # 假设重定位立即发生 (或延迟一周期，这里简化为立即扣除，下一周期到达)
        # 简单起见，假设 T=1 到达。
        # 检查是否有足够的车辆进行重定位
        actual_relocation = np.zeros_like(relocation)
        relocation_cost = 0.0
        
        for i in range(self.num_stations):
            available = self.vehicles[i]
            outgoing = np.sum(relocation[i, :])
            
            if outgoing > available:
                # 车辆不足，按比例缩减
                scale = available / outgoing
                for j in range(self.num_stations):
                    count = int(relocation[i, j] * scale)
                    actual_relocation[i, j] = count
            else:
                actual_relocation[i, :] = relocation[i, :]
                
            # 扣除车辆
            self.vehicles[i] -= np.sum(actual_relocation[i, :])
            
            # 计算成本
            for j in range(self.num_stations):
                if actual_relocation[i, j] > 0:
                    cost = self.loader.relocation_cost[i, j] * actual_relocation[i, j]
                    relocation_cost += cost

        # 2. 处理用户需求 (Customer Demand)
        # 生成随机需求
        # lambda = a - b * p
        # 注意：这里使用实际定价计算 lambda
        a = self.loader.data['a'][:, :, t % 24]
        b = self.loader.data['b'][:, :, t % 24]
        
        demand_rate = np.maximum(0, a - b * pricing)
        realized_demand = np.random.poisson(demand_rate)
        
        revenue = 0.0
        step_lost_sales = 0
        step_total_demand = 0
        
        # 满足需求
        trips = np.zeros((self.num_stations, self.num_stations), dtype=int)
        
        for i in range(self.num_stations):
            available = self.vehicles[i]
            total_req = np.sum(realized_demand[i, :])
            step_total_demand += total_req
            
            if total_req <= available:
                # 满足所有需求
                trips[i, :] = realized_demand[i, :]
                self.vehicles[i] -= total_req
            else:
                # 车辆不足，发生 Lost Sales
                step_lost_sales += (total_req - available)
                # 随机分配或按比例分配可用车辆
                if total_req > 0:
                    probs = realized_demand[i, :] / total_req
                    # 多项分布采样实际出发的车辆
                    served = np.random.multinomial(available, probs)
                    trips[i, :] = served
                    self.vehicles[i] = 0
            
            # 计算收入
            for j in range(self.num_stations):
                revenue += trips[i, j] * pricing[i, j]

        # 4. 充电与电网交互 (Charging & Grid Interaction)
        # 获取 Agent 的充电决策 (单位: kWh)
        # 如果 Agent 没给，默认策略：充满所有在站车辆
        # 限制：充电量不能超过电池容量空缺，也不能超过充电桩功率限制
        
        charging_demand = actions.get('charging', None)
        
        if charging_demand is None:
            # 默认策略：尽可能充满
            max_energy = self.vehicles * config.BATTERY_CAPACITY
            energy_needed = np.maximum(0, max_energy - self.energy)
            # 功率限制 (假设每辆车最大充电功率 CHARGING_RATE kW, 1小时)
            max_power = self.vehicles * config.CHARGING_RATE * config.DELTA_T
            charging_demand = np.minimum(energy_needed, max_power)
        
        # 转换为 MW 用于 OPF (1 MW = 1000 kW)
        charging_load_mw = charging_demand / 1000.0
        
        # 调用 Power Grid 计算 OPF
        success, lmps, gen_cost = self.grid.solve_opf(t, charging_load_mw)
        
        step_charging_cost = 0.0
        if success:
            # 计算充电成本: sum(LMP * ChargingAmount)
            # LMP 单位 $/MWh. ChargingAmount 单位 MWh (charging_load_mw * 1h)
            # 注意：lmps 是 33 维数组，我们需要提取 8 个站点的 LMP
            station_lmps = lmps[self.loader.station_nodes]
            step_charging_cost = np.sum(station_lmps * charging_load_mw)
        else:
            # OPF 失败 (例如电压越限)，施加巨额惩罚
            step_charging_cost = 1e5 
            # 强制充电失败? 或者按惩罚价格计算?
            # 这里简单处理：充电成功但巨贵
        
        # 更新能量状态
        self.energy += charging_demand
        
        # 扣除车辆行驶能耗 (简化：每辆车出发时扣除预计能耗? 或者到达时扣除?)
        # 简单起见：假设车辆出发时必须有足够电量，到达时电量减少
        # 这里先不细扣每辆车的电，只更新总池子
        # 假设平均能耗: 0.2 kWh/km * avg_dist? 
        # 使用 loader.get_energy_consumption(i, j, t) (单位可能是 p.u. 或 kWh)
        # mat['L'] 是能耗。
        # 遍历 trips 和 actual_relocation 扣除能量
        
        energy_consumed = 0.0
        # 用户行程能耗
        for i in range(self.num_stations):
            for j in range(self.num_stations):
                if trips[i, j] > 0:
                    # L[i,j,t] 单位是 p.u. 需要转 kWh? 
                    # 假设 mat['L'] 已经是归一化后的能耗比例? 
                    # 论文: L_ij 是 energy consumption. E_max = 27 kWh.
                    # mat['L'] mean ~ 8e-5. mat['E_max'] ~ 9e-4.
                    # 比例: L / E_max ~ 0.1 (10% 电量). 合理.
                    # 所以消耗电量 = trips * (L_ij / E_max_pu) * BATTERY_CAPACITY
                    
                    l_pu = self.loader.get_energy_consumption(i, j, t)
                    e_max_pu = self.loader.data['E_max_pu']
                    consumption_per_trip = (l_pu / e_max_pu) * config.BATTERY_CAPACITY
                    
                    total_cons = trips[i, j] * consumption_per_trip
                    self.energy[i] -= total_cons # 从出发站扣除 (简化)
                    energy_consumed += total_cons

        # 重定位行程能耗
        for i in range(self.num_stations):
            for j in range(self.num_stations):
                if actual_relocation[i, j] > 0:
                    l_pu = self.loader.get_energy_consumption(i, j, t)
                    e_max_pu = self.loader.data['E_max_pu']
                    consumption_per_trip = (l_pu / e_max_pu) * config.BATTERY_CAPACITY
                    
                    total_cons = actual_relocation[i, j] * consumption_per_trip
                    self.energy[i] -= total_cons
                    energy_consumed += total_cons
        
        # 确保能量非负 (简化处理)
        self.energy = np.maximum(0, self.energy)

        # 5. 车辆到达 (Vehicle Arrival)
        # 更新车辆数和到达车辆的剩余能量
        
        incoming_trips = np.sum(trips, axis=0) # 列求和：所有到 j 的用户车辆
        incoming_reloc = np.sum(actual_relocation, axis=0) # 所有到 j 的重定位车辆
        
        self.vehicles += incoming_trips + incoming_reloc
        
        # 计算到达车辆的剩余能量并加入目标站点能量池
        # 假设车辆出发时是满电 (E_max)，到达时剩余 (E_max - consumption)
        for j in range(self.num_stations):
            incoming_energy = 0.0
            for i in range(self.num_stations):
                # 用户行程到达
                if trips[i, j] > 0:
                    l_pu = self.loader.get_energy_consumption(i, j, t)
                    e_max_pu = self.loader.data['E_max_pu']
                    consumption_per_trip = (l_pu / e_max_pu) * config.BATTERY_CAPACITY
                    remaining_energy = config.BATTERY_CAPACITY - consumption_per_trip
                    incoming_energy += trips[i, j] * remaining_energy
                
                # 重定位到达
                if actual_relocation[i, j] > 0:
                    l_pu = self.loader.get_energy_consumption(i, j, t)
                    e_max_pu = self.loader.data['E_max_pu']
                    consumption_per_trip = (l_pu / e_max_pu) * config.BATTERY_CAPACITY
                    remaining_energy = config.BATTERY_CAPACITY - consumption_per_trip
                    incoming_energy += actual_relocation[i, j] * remaining_energy
            
            self.energy[j] += incoming_energy
        
        # 6. 更新统计
        profit = revenue - relocation_cost - step_charging_cost
        self.total_profit += profit
        self.lost_sales += step_lost_sales
        self.total_demand += step_total_demand
        self.total_charging_cost += step_charging_cost
        
        self.current_step += 1
        done = self.current_step >= self.time_steps
        
        info = {
            'revenue': revenue,
            'relocation_cost': relocation_cost,
            'charging_cost': step_charging_cost,
            'lost_sales': step_lost_sales,
            'demand': step_total_demand,
            'trips': trips,
            'relocation': actual_relocation,
            'lmps': lmps[self.loader.station_nodes] if success else None
        }
        
        return self._get_state(), profit, done, info
