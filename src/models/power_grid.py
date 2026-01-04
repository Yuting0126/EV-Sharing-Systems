import numpy as np
import copy
from pypower.api import runopf, ppoption
from pypower.idx_bus import PD, QD, VM, VA, BUS_I, BUS_TYPE, REF, PV, PQ
from pypower.idx_gen import PG, QG, GEN_BUS, PMAX, PMIN, QMAX, QMIN, GEN_STATUS
# COST_1 is not directly available, use index 4 for c1 in polynomial model (n=2)
# gencost cols: MODEL(0), STARTUP(1), SHUTDOWN(2), NCOST(3), COST(4)...
COST_1 = 4
from . import config
from .case33bw import case33bw

class PowerGrid:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.base_mpc = case33bw() # 加载 IEEE 33 节点标准算例
        self._init_grid_params()
        
    def _init_grid_params(self):
        """
        根据 parameters.mat 更新电网参数
        """
        mpc = self.base_mpc
        
        # 1. 设置发电机 (Distributed Generators)
        # 论文中提到 DG 在节点 4, 9, 14, 19, 24, 29 (索引 3, 8, 13, 18, 23, 28)
        # 注意：case33bw 默认只有一个 Slack Bus (节点0)
        # 我们需要添加额外的发电机行到 mpc['gen']
        
        dg_buses = [3, 8, 13, 18, 23, 28] # 0-indexed
        # mat 文件中的 rho_g 是发电成本
        rho_g = self.loader.data['rho_g'].flatten()
        
        new_gens = []
        new_gencosts = []
        
        # 保留 Slack Bus (第一行)
        slack_gen = mpc['gen'][0].copy()
        slack_cost = mpc['gencost'][0].copy()
        
        # Slack Bus 成本: rho = 0.20 (mat['rho'])
        slack_cost[COST_1] = 0.20 # 线性成本系数 c1
        
        new_gens.append(slack_gen)
        new_gencosts.append(slack_cost)
        
        for bus_idx in dg_buses:
            # 创建新发电机行
            # 格式: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf]
            gen_row = np.zeros_like(slack_gen)
            gen_row[GEN_BUS] = bus_idx + 1 # 1-indexed for pypower
            gen_row[GEN_STATUS] = 1
            gen_row[PMAX] = 0.17 * 100 # p.u. * baseMVA (假设 base=100? 不，通常 base=1 或 10)
            # 论文中 p_gu = 0.17 p.u. 
            # pypower 中 PMAX 单位是 MW。如果 baseMVA=100，则为 17 MW。如果 baseMVA=1，则为 0.17 MW。
            # case33bw 默认 baseMVA = 10.
            # 让我们假设 mat 文件中的 p.u. 是基于 10 MVA 的? 
            # 论文中 p_df 平均 0.03 p.u. -> 0.3 MW? 
            # 让我们统一使用 MW。假设 mat 中的 p.u. 对应 baseMVA=1 (方便起见) 或者 10。
            # 检查 mat['p_df'] 的值: mean=0.03. 
            # IEEE 33 总负荷大约 3.7 MW. 
            # 0.03 * 33 ≈ 1.0. 看起来 baseMVA 可能是 100? 
            # 不纠结，我们直接用 pypower 的默认值，然后把 mat 的 p.u. 转换过去。
            # case33bw 总负荷: sum(PD) = 3.715 MW.
            # mat['p_df'] mean 0.03. sum ~ 1.0. 
            # 看来 mat 里的 p.u. 是基于 100 MVA 的? (0.03 * 100 = 3 MW).
            # 让我们假设 baseMVA = 100.
            
            baseMVA = 100.0
            gen_row[PMAX] = 0.17 * baseMVA * 10 # 增加发电机容量以避免不可行 (原论文可能有多台或参数不同)
            gen_row[PMIN] = 0.0
            gen_row[QMAX] = 0.17 * baseMVA
            gen_row[QMIN] = -0.17 * baseMVA
            gen_row[VM] = 1.0
            
            new_gens.append(gen_row)
            
            # 成本
            cost_row = np.zeros_like(slack_cost)
            cost_row[0] = 2 # polynomial model
            cost_row[3] = 2 # n=2 (linear: c1*P + c0) -> pypower polynomial format: c2*P^2 + c1*P + c0
            # 实际上 pypower 的 polynomial 是 c(n)*P^n + ... + c(0)
            # 线性成本: n=2 (参数个数), c1, c0.
            # 论文是线性成本 rho_g * P
            cost_row[4] = rho_g[bus_idx] * 100 # c1 (放大一点以便观察)
            cost_row[5] = 0.0            # c0
            new_gencosts.append(cost_row)
            
        mpc['gen'] = np.array(new_gens)
        mpc['gencost'] = np.array(new_gencosts)
        self.base_mpc = mpc

    def solve_opf(self, t, charging_loads):
        """
        求解 t 时刻的 OPF
        charging_loads: (8,) 数组，各站点的充电功率 (MW)
        """
        # 使用深拷贝避免修改基础数据
        mpc = copy.deepcopy(self.base_mpc)
        
        # 1. 更新基础负荷 (Time-varying Load)
        # mat['p_df'] 是 (33, 24). 
        # 假设 baseMVA = 100.
        baseMVA = 100.0
        p_df = self.loader.data['p_df'][:, t % 24] * baseMVA
        
        # 更新 bus 矩阵的 PD 列
        # pypower bus 索引从 0 开始? 不，矩阵行索引是 0..32
        mpc['bus'][:, PD] = p_df
        
        # 2. 叠加 EV 充电负荷
        # 映射: station_idx -> grid_node_idx
        station_nodes = self.loader.station_nodes
        
        for i, node_idx in enumerate(station_nodes):
            # 叠加到对应节点的 PD 上
            mpc['bus'][node_idx, PD] += charging_loads[i]
            
        # 3. 运行 OPF
        # 尝试 AC OPF 但修复维度问题? 不，先调试 DC OPF 为什么失败
        # 可能是负荷过大或发电机容量不足
        # 打印一些调试信息
        # print(f"Total Load: {np.sum(mpc['bus'][:, PD])} MW")
        # print(f"Total Gen Cap: {np.sum(mpc['gen'][:, PMAX])} MW")
        
        opt = ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=1) # 开启 DC OPF
        result = runopf(mpc, opt)
        
        success = result['success']
        if success:
            # 提取 LMP (Locational Marginal Price)
            # 在 DC OPF 中是 bus_lam_p, AC OPF 中是 ...
            # pypower runopf 默认是 AC OPF.
            # 节点边际价格通常在 result['bus'][:, LAM_P] (对于 DC)
            # 对于 AC OPF，影子价格比较复杂，通常用 lambda_p
            # pypower 的 result['bus'] 列定义里好像没有直接的 LMP?
            # 实际上 result['bus'] 的 LAM_P 列 (索引 13) 就是 Lagrange multiplier for real power balance
            
            LAM_P = 13
            lmps = result['bus'][:, LAM_P]
            
            # 提取总发电成本
            cost = result['f']
            
            return True, lmps, cost
        else:
            return False, None, None

if __name__ == "__main__":
    from src.data_loader import DataLoader
    loader = DataLoader()
    grid = PowerGrid(loader)
    
    # 测试 t=10, 无充电
    success, lmps, cost = grid.solve_opf(10, np.zeros(8))
    if success:
        print(f"OPF Solved! Cost: ${cost:.2f}")
        print(f"Avg LMP: ${np.mean(lmps):.4f}/MWh")
    else:
        print("OPF Failed.")
