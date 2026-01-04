import scipy.io
import numpy as np
import os
from . import config

class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path or config.DATA_PATH
        self.data = {}
        self.is_synthetic = False  # 标记是否为合成数据集
        self._load_data()
        self._process_derived_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        mat = scipy.io.loadmat(self.file_path)
        
        # 检测是否为合成数据集 (通过检查是否有 lmp 字段)
        self.is_synthetic = 'lmp' in mat
        
        # 提取关键矩阵
        self.data['a'] = mat['a']  # 需求截距 (NS, NS, T)
        self.data['b'] = mat['b']  # 价格弹性 (NS, NS, T)
        self.data['L'] = mat['L']  # 能耗矩阵 (NS, NS, T)
        
        self.data['N_0'] = mat['N_0'].flatten().astype(int)  # 初始车辆
        self.data['N_u'] = mat['N_u'].flatten().astype(int)  # 容量上限
        
        if self.is_synthetic:
            # 合成数据集 - 直接使用物理单位
            self.data['E_max'] = float(mat['E_max'][0][0])  # kWh
            self.data['E_min'] = float(mat['E_min'][0][0])  # kWh
            self.data['E_max_pu'] = self.data['E_max'] / config.BATTERY_CAPACITY
            self.data['E_min_pu'] = self.data['E_min'] / config.BATTERY_CAPACITY
            
            # LMP 电价曲线
            self.data['lmp'] = mat['lmp'].flatten()  # (T,) $/MWh
            
            # 重定位成本矩阵 (如果存在)
            if 'gamma' in mat:
                self.data['gamma'] = mat['gamma']
            
            # 站点坐标 (如果存在)
            if 'station_coords' in mat:
                self.data['station_coords'] = mat['station_coords']
            if 'station_type_codes' in mat:
                self.data['station_type_codes'] = mat['station_type_codes'].flatten()
        else:
            # 原始数据集 - 使用标幺值
            self.data['E_max_pu'] = mat['E_max'][0][0]
            self.data['E_min_pu'] = mat['E_min'][0][0]
            self.data['E_max'] = config.BATTERY_CAPACITY
            self.data['E_min'] = config.MIN_BATTERY
        
        # 电网数据
        self.data['p_df'] = mat['p_df']  # 基础负荷 (33, T)
        self.data['rho_g'] = mat['rho_g']  # 发电成本

    def _process_derived_data(self):
        """生成项目中缺失但需要的衍生数据"""
        
        # 1. 生成重定位成本矩阵 (Relocation Cost)
        # 使用平均能耗作为距离的代理
        avg_L = np.mean(self.data['L'], axis=2) # (8, 8)
        self.relocation_cost = config.RELOCATION_BASE_COST + \
                               config.RELOCATION_DIST_COST_FACTOR * avg_L
        
        # 2. 生成泊松分布的到达率 lambda (Demand Rate)
        # lambda = max(0, a - b * p_ref)
        # 假设参考价格 p_ref = 1.0 (单位价格)
        p_ref = config.BASE_PRICE
        self.demand_rate = np.maximum(0, self.data['a'] - self.data['b'] * p_ref)
        
        # 3. 站点映射
        self.station_nodes = np.array(config.STATION_NODE_MAPPING)

    def get_demand_rate(self, t):
        """获取 t 时刻的 OD 需求率矩阵 (8, 8)"""
        t_idx = t % 24
        return self.demand_rate[:, :, t_idx]

    def get_energy_consumption(self, i, j, t):
        """获取 t 时刻 i->j 的能耗"""
        t_idx = t % 24
        return self.data['L'][i, j, t_idx]

    def get_initial_state(self):
        """返回初始车辆数和初始能量 (单位: kWh)"""
        # 使用物理单位 kWh，而非标幺值
        initial_energy = self.data['N_0'] * config.BATTERY_CAPACITY * 0.8  # 假设初始 80% 电量
        return self.data['N_0'].copy(), initial_energy

if __name__ == "__main__":
    loader = DataLoader()
    print("Data loaded successfully.")
    print(f"Demand Rate Shape: {loader.demand_rate.shape}")
    print(f"Relocation Cost Sample (0->1): {loader.relocation_cost[0, 1]:.4f}")
