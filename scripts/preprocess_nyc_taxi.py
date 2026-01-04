"""
preprocess_nyc_taxi.py
======================
NYC Taxi 数据预处理脚本

功能:
1. 读取原始 Parquet 文件
2. 筛选曼哈顿区域的行程
3. 将经纬度聚类为 N 个站点 (Zone)
4. 按时段统计 OD 矩阵 (Origin-Destination)
5. 输出可直接用于实验的 CSV 文件

依赖:
    pip install pandas pyarrow numpy scikit-learn

作者: [Your Name]
日期: 2024-12
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import os

# ============================================================
# 配置参数
# ============================================================

INPUT_FILE = "data/yellow_tripdata_2019-01.parquet"
OUTPUT_DIR = "data/processed"
NUM_ZONES = 10  # 聚类站点数量

# 曼哈顿区域边界 (经纬度)
MANHATTAN_BOUNDS = {
    'lat_min': 40.70,
    'lat_max': 40.82,
    'lon_min': -74.02,
    'lon_max': -73.93
}

# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("NYC Taxi Data Preprocessing")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: 读取数据
    print("\n[Step 1] Loading raw data...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Loaded {len(df):,} records")
    print(f"  Columns: {list(df.columns)}")
    
    # Step 2: 筛选曼哈顿区域
    print("\n[Step 2] Filtering Manhattan trips...")
    
    # 2019年及以后的数据使用 LocationID 而非经纬度
    if 'pickup_latitude' in df.columns:
        # 旧版数据 (有经纬度)
        df = df[
            (df['pickup_latitude'] >= MANHATTAN_BOUNDS['lat_min']) &
            (df['pickup_latitude'] <= MANHATTAN_BOUNDS['lat_max']) &
            (df['pickup_longitude'] >= MANHATTAN_BOUNDS['lon_min']) &
            (df['pickup_longitude'] <= MANHATTAN_BOUNDS['lon_max']) &
            (df['dropoff_latitude'] >= MANHATTAN_BOUNDS['lat_min']) &
            (df['dropoff_latitude'] <= MANHATTAN_BOUNDS['lat_max']) &
            (df['dropoff_longitude'] >= MANHATTAN_BOUNDS['lon_min']) &
            (df['dropoff_longitude'] <= MANHATTAN_BOUNDS['lon_max'])
        ].copy()
        print(f"  Remaining: {len(df):,} records")
        use_location_id = False
    else:
        # 新版数据 (使用 LocationID)
        print("  Note: Using LocationID (2019+ format)")
        # 曼哈顿区域的 LocationID 列表
        manhattan_loc_ids = [4, 12, 13, 79, 87, 88, 100, 113, 114, 125, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]
        df = df[
            (df['PULocationID'].isin(manhattan_loc_ids)) &
            (df['DOLocationID'].isin(manhattan_loc_ids))
        ].copy()
        print(f"  Remaining after LocationID filter: {len(df):,} records")
        use_location_id = True
    
    # Step 3: 生成 Zone 映射
    print("\n[Step 3] Generating zone mapping...")
    if use_location_id:
        # 对 LocationID 进行聚类映射
        unique_pu = df['PULocationID'].unique()
        unique_do = df['DOLocationID'].unique()
        all_locs = np.union1d(unique_pu, unique_do)
        
        # 映射: 将 LocationID 映射到 0 ~ NUM_ZONES-1
        loc_to_zone = {loc: i % NUM_ZONES for i, loc in enumerate(sorted(all_locs))}
        df['origin_zone'] = df['PULocationID'].map(loc_to_zone)
        df['dest_zone'] = df['DOLocationID'].map(loc_to_zone)
        print(f"  Mapped {len(all_locs)} LocationIDs to {NUM_ZONES} zones")
    else:
        # Step 3: 经纬度聚类
        print("\n[Step 3] Clustering pickup/dropoff locations...")
        
        # 聚类上车点
        pickup_coords = df[['pickup_longitude', 'pickup_latitude']].values
        kmeans_pu = KMeans(n_clusters=NUM_ZONES, random_state=42, n_init=10)
        df['origin_zone'] = kmeans_pu.fit_predict(pickup_coords)
        
        # 聚类下车点
        dropoff_coords = df[['dropoff_longitude', 'dropoff_latitude']].values
        kmeans_do = KMeans(n_clusters=NUM_ZONES, random_state=42, n_init=10)
        df['dest_zone'] = kmeans_do.fit_predict(dropoff_coords)
        
        # 保存聚类中心
        centers = pd.DataFrame({
            'zone_id': range(NUM_ZONES),
            'lon': kmeans_pu.cluster_centers_[:, 0],
            'lat': kmeans_pu.cluster_centers_[:, 1]
        })
        centers.to_csv(f"{OUTPUT_DIR}/zone_centers.csv", index=False)
        print(f"  Saved zone centers to {OUTPUT_DIR}/zone_centers.csv")
    
    # Step 4: 提取时段
    print("\n[Step 4] Extracting time periods...")
    # 使用上车时间
    time_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else 'pickup_datetime'
    df['hour'] = pd.to_datetime(df[time_col]).dt.hour
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    
    # Step 5: 统计 OD 矩阵 (连续 7 天)
    print("\n[Step 5] Computing OD matrices for 7 consecutive days...")
    
    # 选取连续 7 天 (2019-01-15 到 2019-01-21)
    from datetime import timedelta
    start_date = datetime(2019, 1, 15).date()
    num_days = 7
    
    all_days_od = {}
    
    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        df_day = df[df['date'] == current_date]
        print(f"  Day {day_idx + 1}: {current_date} ({current_date.strftime('%A')}), Records: {len(df_day):,}")
        
        # 计算该天每小时的 OD 矩阵
        od_matrices = {}
        for hour in range(24):
            df_hour = df_day[df_day['hour'] == hour]
            od = df_hour.groupby(['origin_zone', 'dest_zone']).size().unstack(fill_value=0)
            od = od.reindex(index=range(NUM_ZONES), columns=range(NUM_ZONES), fill_value=0)
            od_matrices[hour] = od.values
        
        # 合并为 3D 数组
        od_tensor = np.stack([od_matrices[h] for h in range(24)], axis=2)  # (NS, NS, 24)
        all_days_od[day_idx] = od_tensor
        
        # 保存单日文件
        date_str = current_date.strftime('%Y%m%d')
        np.save(f"{OUTPUT_DIR}/od_matrix_{date_str}.npy", od_tensor)
    
    # 保存汇总文件 (7天合并为 4D 数组)
    od_week = np.stack([all_days_od[d] for d in range(num_days)], axis=3)  # (NS, NS, 24, 7)
    np.save(f"{OUTPUT_DIR}/od_matrix_week.npy", od_week)
    print(f"\n  Saved weekly OD tensor shape {od_week.shape} to {OUTPUT_DIR}/od_matrix_week.npy")
    
    # Step 6: 统计摘要
    print("\n[Step 6] Summary Statistics:")
    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        daily_trips = all_days_od[day_idx].sum()
        print(f"  Day {day_idx + 1} ({current_date.strftime('%a')}): {daily_trips:,.0f} trips")
    print(f"  Total week trips: {od_week.sum():,.0f}")
    
    print("\n[Preprocessing Complete]")
    print(f"  Output files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
