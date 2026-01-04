# System Configuration & Mapping
# 这个文件定义了交通网(Traffic)与电网(Grid)的连接关系，以及基本参数

# ==========================================
# 1. 节点映射 (Node Mapping)
# ==========================================
# 格式: {Traffic_Zone_ID: IEEE_Grid_Bus_ID}
# 这里的 Traffic Zone ID 对应 NYC Taxi 聚类后的区域 ID
# 这里的 Grid Bus ID 对应 IEEE 33 节点的编号 (1-33)
# 注意：Bus 1 是平衡节点(变电站)，通常不接充电负荷，所以选下游节点
TRAFFIC_TO_GRID_MAPPING = {
    1: 4,   # 比如：曼哈顿下城 对应 电网 Bus 4
    2: 7,   # 比如：时代广场 对应 电网 Bus 7
    3: 8,
    4: 14,
    5: 18,  # 可能是一个负荷中心
    6: 22,
    7: 24,
    8: 25,
    9: 29,
    10: 32  # 比如：上东区
}

# ==========================================
# 2. 系统参数 (System Parameters)
# ==========================================
TIME_STEPS = 24              # 一天24小时
DELTA_T = 1.0               # 时间步长 (小时)

# EV Parameters
EV_BATTERY_CAPACITY = 27.0  # kWh (Nissan Leaf 早期型号)
EV_MIN_SOC = 0.1            # 最小剩余电量 10%
EV_CHARGING_SPEED = 7.0     # kW (慢充 L2)

# Price Sensitivity (Demand Function)
# D = a - b * Price
DEMAND_ELASTICITY_B = 20.0  

# ==========================================
# 3. 路径配置 (Path Config)
# ==========================================
DATA_PATH = "./data"
IEEE_BUS_FILE = "ieee33_bus.csv"
IEEE_BRANCH_FILE = "ieee33_branch.csv"
