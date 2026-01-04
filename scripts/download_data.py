"""
真实数据集下载器
============================
下载公开的共享出行数据和电价数据

数据源：
1. Citi Bike NYC - 共享单车行程数据
2. Capital Bikeshare DC - 共享单车行程数据
3. PJM - 电网 LMP 数据

作者：研究团队
日期：2025年12月
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 数据下载目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'real_data')

def ensure_data_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return DATA_DIR


class CitiBikeDownloader:
    """
    Citi Bike NYC 数据下载器
    
    数据源：https://citibikenyc.com/system-data
    包含：行程开始/结束时间、站点、用户类型等
    """
    
    BASE_URL = "https://s3.amazonaws.com/tripdata"
    
    @staticmethod
    def get_available_months():
        """获取可下载的月份列表"""
        # 2024年数据
        months = []
        for year in [2023, 2024]:
            for month in range(1, 13):
                months.append(f"{year}{month:02d}")
        return months[-12:]  # 最近12个月
    
    @staticmethod
    def download_month(year_month, save_dir=None):
        """
        下载指定月份的数据
        
        Parameters:
        -----------
        year_month : str
            格式 "YYYYMM"，如 "202401"
        save_dir : str
            保存目录
        """
        if save_dir is None:
            save_dir = ensure_data_dir()
            
        # 新格式文件名 (2024年后)
        filename = f"{year_month}-citibike-tripdata.csv.zip"
        url = f"{CitiBikeDownloader.BASE_URL}/{filename}"
        
        output_path = os.path.join(save_dir, f"citibike_{year_month}.csv")
        
        if os.path.exists(output_path):
            print(f"[跳过] {year_month} 数据已存在")
            return output_path
            
        print(f"[下载] Citi Bike {year_month}...")
        print(f"       URL: {url}")
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                # 解压 ZIP
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        # 只保留需要的列
                        cols_to_keep = [
                            'started_at', 'ended_at',
                            'start_station_name', 'start_station_id',
                            'end_station_name', 'end_station_id',
                            'start_lat', 'start_lng',
                            'end_lat', 'end_lng'
                        ]
                        available_cols = [c for c in cols_to_keep if c in df.columns]
                        df = df[available_cols]
                        df.to_csv(output_path, index=False)
                        print(f"       ✅ 已保存 {len(df)} 条记录")
                        return output_path
            else:
                print(f"       ❌ 下载失败: HTTP {response.status_code}")
                # 尝试旧格式
                return CitiBikeDownloader._try_old_format(year_month, save_dir)
        except Exception as e:
            print(f"       ❌ 错误: {e}")
            return None
    
    @staticmethod
    def _try_old_format(year_month, save_dir):
        """尝试旧格式文件名"""
        filename = f"JC-{year_month}-citibike-tripdata.csv.zip"
        url = f"{CitiBikeDownloader.BASE_URL}/{filename}"
        
        print(f"       尝试旧格式: {filename}")
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                output_path = os.path.join(save_dir, f"citibike_{year_month}.csv")
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        df.to_csv(output_path, index=False)
                        print(f"       ✅ 已保存 {len(df)} 条记录")
                        return output_path
        except:
            pass
        return None


class CapitalBikeshareDownloader:
    """
    Capital Bikeshare (Washington DC) 数据下载器
    
    数据源：https://capitalbikeshare.com/system-data
    """
    
    BASE_URL = "https://s3.amazonaws.com/capitalbikeshare-data"
    
    @staticmethod
    def download_month(year_month, save_dir=None):
        """下载指定月份数据"""
        if save_dir is None:
            save_dir = ensure_data_dir()
            
        filename = f"{year_month}-capitalbikeshare-tripdata.zip"
        url = f"{CapitalBikeshareDownloader.BASE_URL}/{filename}"
        
        output_path = os.path.join(save_dir, f"capitalbikeshare_{year_month}.csv")
        
        if os.path.exists(output_path):
            print(f"[跳过] {year_month} 数据已存在")
            return output_path
            
        print(f"[下载] Capital Bikeshare {year_month}...")
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        df.to_csv(output_path, index=False)
                        print(f"       ✅ 已保存 {len(df)} 条记录")
                        return output_path
            else:
                print(f"       ❌ 下载失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"       ❌ 错误: {e}")
        return None


class PJMDataDownloader:
    """
    PJM 电网 LMP 数据下载器
    
    数据源：https://dataminer2.pjm.com
    """
    
    @staticmethod
    def get_sample_lmp_data():
        """
        获取示例 LMP 数据
        
        由于 PJM API 需要注册，这里生成基于真实模式的模拟数据
        """
        print("[生成] PJM LMP 示例数据...")
        
        # 典型 PJM LMP 模式 (基于历史数据统计)
        # 单位：$/MWh
        typical_lmp_pattern = np.array([
            25.3, 23.1, 21.5, 20.8, 21.2, 24.5,  # 0-5: 夜间低谷
            32.1, 45.6, 52.3, 48.7, 42.1, 38.5,  # 6-11: 早高峰
            36.2, 35.8, 37.1, 42.5, 55.8, 62.3,  # 12-17: 午后上升
            58.7, 48.2, 38.6, 32.1, 28.5, 26.2   # 18-23: 晚高峰下降
        ])
        
        # 添加随机波动
        np.random.seed(42)
        noise = np.random.normal(0, 3, 24)
        lmp = typical_lmp_pattern + noise
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'hour': range(24),
            'lmp_$/MWh': lmp,
            'source': 'PJM_simulated_based_on_historical'
        })
        
        save_dir = ensure_data_dir()
        output_path = os.path.join(save_dir, 'pjm_lmp_sample.csv')
        df.to_csv(output_path, index=False)
        
        print(f"       ✅ 已保存到 {output_path}")
        print(f"       电价范围: ${lmp.min():.2f} ~ ${lmp.max():.2f} /MWh")
        
        return df


def convert_bikeshare_to_ev_demand(csv_path, num_stations=8, output_name='demand_matrix.npy'):
    """
    将共享单车数据转换为电动车需求矩阵
    
    转换逻辑：
    1. 聚合为小时级 OD 矩阵
    2. 选择最繁忙的 N 个站点
    3. 生成 a[i,j,t] 需求矩阵
    """
    print(f"\n[转换] 将 {os.path.basename(csv_path)} 转换为 EV 需求矩阵...")
    
    df = pd.read_csv(csv_path)
    
    # 解析时间
    time_col = 'started_at' if 'started_at' in df.columns else 'Start date'
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    
    # 找出最繁忙的站点
    start_col = 'start_station_id' if 'start_station_id' in df.columns else 'Start station number'
    end_col = 'end_station_id' if 'end_station_id' in df.columns else 'End station number'
    
    # 统计站点使用频率
    station_counts = pd.concat([
        df[start_col].value_counts(),
        df[end_col].value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False)
    
    top_stations = station_counts.head(num_stations).index.tolist()
    
    # 筛选包含这些站点的行程
    df_filtered = df[
        df[start_col].isin(top_stations) & 
        df[end_col].isin(top_stations)
    ].copy()
    
    # 创建站点索引映射
    station_map = {s: i for i, s in enumerate(top_stations)}
    df_filtered['origin'] = df_filtered[start_col].map(station_map)
    df_filtered['destination'] = df_filtered[end_col].map(station_map)
    
    # 生成 OD 矩阵 [origin, destination, hour]
    demand_matrix = np.zeros((num_stations, num_stations, 24))
    
    for _, row in df_filtered.iterrows():
        if pd.notna(row['origin']) and pd.notna(row['destination']):
            i, j, t = int(row['origin']), int(row['destination']), int(row['hour'])
            demand_matrix[i, j, t] += 1
    
    # 归一化（日均需求）
    num_days = (pd.to_datetime(df[time_col]).max() - 
                pd.to_datetime(df[time_col]).min()).days + 1
    demand_matrix = demand_matrix / max(num_days, 1)
    
    # 放大到 EV 规模（假设 EV 需求是单车的 2 倍）
    demand_matrix = demand_matrix * 2
    
    save_dir = ensure_data_dir()
    output_path = os.path.join(save_dir, output_name)
    np.save(output_path, demand_matrix)
    
    print(f"       ✅ 需求矩阵形状: {demand_matrix.shape}")
    print(f"       ✅ 日均总需求: {demand_matrix.sum():.0f} 人次")
    print(f"       ✅ 已保存到: {output_path}")
    
    return demand_matrix


def main():
    """主函数"""
    print("=" * 60)
    print("真实数据集下载器")
    print("=" * 60)
    
    # 1. 下载 Citi Bike 数据 (尝试最近一个月)
    print("\n" + "-" * 40)
    print("1. Citi Bike NYC 数据")
    print("-" * 40)
    
    # 尝试下载 2024年10月数据
    citibike_path = CitiBikeDownloader.download_month("202410")
    
    # 2. 下载 Capital Bikeshare 数据
    print("\n" + "-" * 40)
    print("2. Capital Bikeshare DC 数据")
    print("-" * 40)
    
    capital_path = CapitalBikeshareDownloader.download_month("202410")
    
    # 3. 生成 PJM LMP 数据
    print("\n" + "-" * 40)
    print("3. PJM LMP 电价数据")
    print("-" * 40)
    
    lmp_df = PJMDataDownloader.get_sample_lmp_data()
    
    # 4. 转换为 EV 需求矩阵
    print("\n" + "-" * 40)
    print("4. 转换为 EV 需求矩阵")
    print("-" * 40)
    
    if citibike_path and os.path.exists(citibike_path):
        demand = convert_bikeshare_to_ev_demand(citibike_path)
    elif capital_path and os.path.exists(capital_path):
        demand = convert_bikeshare_to_ev_demand(capital_path)
    else:
        print("       ⚠️ 无数据可转换，请手动下载")
    
    # 总结
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)
    print(f"\n数据保存目录: {DATA_DIR}")
    print("\n可用数据文件:")
    
    for f in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    print("\n" + "-" * 40)
    print("使用建议:")
    print("-" * 40)
    print("1. 需求数据 → 用于验证定价算法和重定位策略")
    print("2. LMP 数据 → 用于验证充电调度算法")
    print("3. 结合使用 → 完整验证 'Operational Trilemma' 框架")


if __name__ == '__main__':
    main()

