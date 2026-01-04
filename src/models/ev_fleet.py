import gurobipy as gp
from gurobipy import GRB
import numpy as np
from . import config

class MIQPAgent:
    """
    实现报告中的 Algorithm 1: Integrated Mixed-Integer Quadratic Programming (MIQP)
    作为一个滚动时域控制器 (MPC) 的一步决策。
    
    注意：完整的 MIQP 是针对全天 24 小时的全局优化。
    在仿真中，我们可以在每个时间步 t，求解从 t 到 T 的剩余问题。
    为了速度，这里简化为只看未来几步 (Look-ahead) 或者单步优化。
    """
    def __init__(self, data_loader, look_ahead=24):
        self.loader = data_loader
        self.look_ahead = look_ahead # 预测未来多少个时间步
        
        # 初始化电价预测 (Base Time-of-Use Curve)
        # 0-6: Low, 7-10: High, 11-16: Med, 17-21: Very High, 22-23: Med
        self.base_lmp_curve = np.array([
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03, # 0-5
            0.04, 0.08, 0.08, 0.08, 0.08,       # 6-10
            0.06, 0.06, 0.06, 0.06, 0.06, 0.06, # 11-16
            0.10, 0.12, 0.12, 0.12, 0.10,       # 17-21
            0.05, 0.04                          # 22-23
        ])
        # 当前预测曲线 (会随观测更新)
        self.predicted_lmps = self.base_lmp_curve.copy()

    def update_belief(self, t, observed_avg_lmp):
        """
        根据观测到的真实 LMP 更新未来的预测。
        使用简单的偏差校正 (Bias Correction)。
        """
        if t >= 24: return
        
        # 计算预测偏差
        predicted = self.predicted_lmps[t]
        bias = observed_avg_lmp - predicted
        
        # 更新未来时刻的预测
        # 假设偏差具有持续性，将未来所有时刻的预测值加上偏差的一部分
        # alpha: 学习率/平滑因子
        alpha = 0.5 
        
        # 只更新 t 之后的时刻
        if t + 1 < 24:
            self.predicted_lmps[t+1:] += alpha * bias
            
            # 确保价格非负且在合理范围内
            self.predicted_lmps[t+1:] = np.clip(self.predicted_lmps[t+1:], 0.01, 0.50)

    def get_action(self, state):
        t_start = state['t']
        current_vehicles = state['vehicles']
        
        # 如果已经到了最后一步，不做决策
        if t_start >= config.TIME_PERIODS:
            return {'pricing': np.ones((8, 8)) * config.BASE_PRICE, 'relocation': np.zeros((8, 8), dtype=int)}

        # 确定优化范围
        t_end = min(config.TIME_PERIODS, t_start + self.look_ahead)
        horizon = t_end - t_start
        
        # 创建模型
        m = gp.Model("EV_Sharing_MPC")
        m.setParam('OutputFlag', 0) # 关闭输出
        
                # --- 变量定义 ---
        # 集合: i, j (站点), k (时间步 0..horizon-1)
        stations = range(config.NUM_STATIONS)
        steps = range(horizon)
        
        # c[i,j,k]: 定价 (连续变量)
        c = m.addVars(stations, stations, steps, lb=0.0, ub=5.0, name="price")
        
        # f[i,j,k]: 实际服务流量 (连续变量)
        # f <= Demand
        f = m.addVars(stations, stations, steps, lb=0.0, name="flow")
        
        # r[i,j,k]: 重定位流量 (整数变量)
        r = m.addVars(stations, stations, steps, vtype=GRB.INTEGER, lb=0.0, name="reloc")
        
        # N[i,k]: 站点车辆数 (状态变量)
        # 容量上限从 loader 读取
        N_u = self.loader.data['N_u']
        N = m.addVars(stations, steps, lb=0.0, name="inventory") 
        for i in stations:
            for k in steps:
                N[i,k].UB = N_u[i]

        # E[i,k]: 站点总能量库存 (kWh)
        # e[i,k]: 站点充电能量 (kWh)
        E = m.addVars(stations, steps, lb=0.0, name="energy_inventory")
        e = m.addVars(stations, steps, lb=0.0, name="charging_energy")
        
        # --- 目标函数 ---
        # Maximize Profit = Revenue - Relocation Cost - Charging Cost - Lost Sales Penalty
        # Revenue = c * f (但 f 受需求约束 f <= a - b*c)
        # Lost Sales Penalty = PENALTY * (Demand - f)
        # Relocation Cost = sum(cost * r)
        # Charging Cost = sum(LMP * e)
        
        PENALTY = config.LOST_SALES_PENALTY
        
        obj = 0
        for k in steps:
            t_real = t_start + k
            # 获取当前时刻的参数
            a = self.loader.data['a'][:, :, t_real % 24]
            b = self.loader.data['b'][:, :, t_real % 24]
            
            # 预测电价 (LMP)
            # 使用更新后的预测曲线
            pred_lmp = self.predicted_lmps[t_real % 24]
            
            for i in stations:
                # Charging Cost
                obj -= pred_lmp * e[i,k]

                for j in stations:
                    # Revenue term: a*c - b*c^2 (由需求函数推导)
                    revenue_potential = a[i,j] * c[i,j,k] - b[i,j] * c[i,j,k] * c[i,j,k]
                    
                    # Lost Sales Penalty term
                    # Lost = (a - b*c) - f
                    # Term = - PENALTY * Lost = - PENALTY * (a - b*c) + PENALTY * f
                    lost_penalty_term = - PENALTY * (a[i,j] - b[i,j] * c[i,j,k]) + PENALTY * f[i,j,k]
                    
                    obj += revenue_potential + lost_penalty_term
                    
                    # Relocation Cost
                    cost_ij = self.loader.relocation_cost[i, j]
                    obj -= cost_ij * r[i,j,k]
        
        m.setObjective(obj, GRB.MAXIMIZE)
        
        # --- 约束条件 ---
        
        # 1. 需求约束 (Demand Function)
        # f[i,j,k] <= a - b * c
        # 且 f >= 0 (由变量定义保证)
        for k in steps:
            t_real = t_start + k
            a = self.loader.data['a'][:, :, t_real % 24]
            b = self.loader.data['b'][:, :, t_real % 24]
            for i in stations:
                for j in stations:
                    m.addConstr(f[i,j,k] <= a[i,j] - b[i,j] * c[i,j,k])
                    # 逻辑约束：如果价格太高导致需求为负，f应为0。
                    # Gurobi 处理 f <= negative 会导致不可行，除非 f 允许负数。
                    # 但 f 定义为 >= 0。所以当 a - b*c < 0 时，约束变成 f <= 负数 -> 不可行。
                    # 修正：我们需要 c <= a/b，即价格不能高到让需求为负。
                    if b[i,j] > 0:
                        m.addConstr(c[i,j,k] <= a[i,j] / b[i,j])

        # 2. 流量守恒 (Flow Conservation / Inventory Dynamics)
        # N[i, k+1] = N[i, k] + Inflow - Outflow
        
        # 3. 能量守恒 (Energy Conservation)
        # E[i, k] = E[i, k-1] + Charging - Consumption_Out + Consumption_In
        # 简化：E[i, k] 是第 k 步结束时的能量
        
        current_energy = state['energy'] # 初始能量状态
        E_max_per_vehicle = config.BATTERY_CAPACITY
        
        # *** 简化能量模型 ***
        # 采用方案 B: 只追踪能量池，不强制满电出发
        # 这样约束更容易满足，优化更稳定
        
        for i in stations:
            # --- k=0 ---
            prev_N = current_vehicles[i]
            prev_E = current_energy[i]
            
            # Clamp prev_E to avoid infeasibility
            if prev_E > prev_N * E_max_per_vehicle:
                prev_E = prev_N * E_max_per_vehicle
            
            outflow = gp.quicksum(f[i,j,0] for j in stations) + gp.quicksum(r[i,j,0] for j in stations)
            
            # 简化能量模型：只计算行程能耗，不强制满电出发
            # 计算平均每辆车的能耗 (使用平均值简化)
            avg_consumption = 0.0
            for j in stations:
                l_pu = self.loader.get_energy_consumption(i, j, t_start)
                cons_kwh = (l_pu / self.loader.data['E_max_pu']) * E_max_per_vehicle
                avg_consumption += cons_kwh
            avg_consumption /= config.NUM_STATIONS  # 平均能耗
            
            # 能量约束：只扣除行程消耗的能量
            energy_consumed = outflow * avg_consumption
            
            m.addConstr(N[i, 0] == prev_N - outflow)
            m.addConstr(E[i, 0] >= 0)
            m.addConstr(E[i, 0] <= N[i, 0] * E_max_per_vehicle)
            
            # 充电功率约束：基于初始车辆数（已知值）
            m.addConstr(e[i, 0] <= prev_N * config.CHARGING_RATE * config.DELTA_T)
            
            # 能量动态：充电增加，行程消耗减少
            m.addConstr(E[i, 0] <= prev_E + e[i, 0] - energy_consumed)

        for k in range(1, horizon):
            t_real = t_start + k
            for i in stations:
                # Outflow at k
                outflow = gp.quicksum(f[i,j,k] for j in stations) + gp.quicksum(r[i,j,k] for j in stations)
                
                # Inflow from k-1
                inflow_trips = gp.quicksum(f[j,i,k-1] for j in stations)
                inflow_reloc = gp.quicksum(r[j,i,k-1] for j in stations)
                
                # 简化能量模型：计算平均能耗
                avg_consumption_out = 0.0
                avg_remaining_in = 0.0
                for j in stations:
                    # i->j 的能耗 (出发)
                    l_pu_out = self.loader.get_energy_consumption(i, j, t_real)
                    cons_kwh_out = (l_pu_out / self.loader.data['E_max_pu']) * E_max_per_vehicle
                    avg_consumption_out += cons_kwh_out
                    
                    # j->i 的剩余能量 (到达)
                    l_pu_in = self.loader.get_energy_consumption(j, i, t_real-1)
                    cons_kwh_in = (l_pu_in / self.loader.data['E_max_pu']) * E_max_per_vehicle
                    avg_remaining_in += (E_max_per_vehicle * 0.8 - cons_kwh_in)  # 假设出发时 80% 电量
                    
                avg_consumption_out /= config.NUM_STATIONS
                avg_remaining_in /= config.NUM_STATIONS
                
                # 车辆守恒
                m.addConstr(N[i, k] == N[i, k-1] - outflow + inflow_trips + inflow_reloc)
                
                # 能量约束 (不等式，更灵活)
                m.addConstr(N[i, k] >= 0)
                m.addConstr(E[i, k] >= 0)
                m.addConstr(E[i, k] <= N[i, k] * E_max_per_vehicle)
                m.addConstr(e[i, k] <= N[i, k] * config.CHARGING_RATE * config.DELTA_T)

        # --- 求解 ---
        m.optimize()
        
        # --- 提取结果 ---
        if m.status == GRB.OPTIMAL:
            # 提取 k=0 的决策
            pricing_action = np.zeros((8, 8))
            reloc_action = np.zeros((8, 8), dtype=int)
            charging_action = np.zeros(8) # 新增
            
            for i in stations:
                charging_action[i] = e[i,0].X
                for j in stations:
                    pricing_action[i,j] = c[i,j,0].X
                    reloc_action[i,j] = int(r[i,j,0].X)
            
            return {'pricing': pricing_action, 'relocation': reloc_action, 'charging': charging_action}
        else:
            print("Optimization failed or infeasible. Fallback to default.")
            return {'pricing': np.ones((8, 8)) * config.BASE_PRICE, 'relocation': np.zeros((8, 8), dtype=int), 'charging': np.zeros(8)}
