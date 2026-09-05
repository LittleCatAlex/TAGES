import pandas as pd
import numpy as np
from pyproj import Transformer
import os
import requests

from data_fetcher import fetch_tvm_data
from drill_processor import advanced_geology_encoder, map_interval_data, parse_json_strata_to_dataframe
from velocity_processor import process_velocity

def twd_to_wgs84(n: float, e: float, datum: str = "TWD97"):
    """將台灣二度分帶座標 (N, E) 轉換為 WGS84 (緯度, 經度)"""
    datum_str = str(datum).upper()
    if "TWD67" in datum_str:
        # EPSG:3828 為 TWD67 TM2
        transformer = Transformer.from_crs("epsg:3828", "epsg:4326", always_xy=True)
    else:
        # EPSG:3826 為 TWD97 TM2
        transformer = Transformer.from_crs("epsg:3826", "epsg:4326", always_xy=True)
    
    lon, lat = transformer.transform(e, n)
    return lat, lon


def fetch_point_seismic(lat: float, lon: float, max_depth: float, output_dir: str = "data_input/seismic"):
    """
    利用你寫好的 fetch_tvm_data，以該點座標為中心微幅擴展成小範圍框，
    去中研院 API 抓取對應位置的震波剖面資料。
    """
    # 因為中研院 API 需要矩形範圍 (lat1, lon1, lat2, lon2)，
    # 我們以鑽孔點為中心，微幅擴展 ±0.01 度來框選該點周邊的網格資料
    offset = 0.01
    lat1 = lat - offset
    lat2 = lat + offset
    lon1 = lon - offset
    lon2 = lon + offset
    
    # 呼叫你實作好的真實 API 抓取資料並存成 CSV
    success = fetch_tvm_data(lat1, lon1, lat2, lon2, depth=int(max_depth), output_dir=output_dir)
    
    if success:
        csv_path = os.path.join(output_dir, "TVM_VerticalProfile_Output.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
            
    raise RuntimeError(f"無法取得座標 ({lat}, {lon}) 的震波資料")

def process_single_well_dynamic(well_name: str, strata_list: list, n: float, e: float, datum: str, step: float = 0.5):
    """將單一鑽孔的文字資料網格化，並動態掛載專屬震波，輸出 (N, 4) 矩陣"""
    
    # 1. 取得最大深度並建立網格
    max_depth = float(strata_list[-1]['depth_m']) if strata_list else 10.0
    target_depths = np.arange(0.0, max_depth + step, step)
    merged_df = pd.DataFrame({'Depth': target_depths})
    
    # 2. 轉換座標並獲取震波資料
    lat, lon = twd_to_wgs84(n, e, datum)
    seismic_df = fetch_point_seismic(lat, lon, max_depth)
    
    # 使用你寫好的 process_velocity 進行震波內插對齊
    seismic_aligned = process_velocity(seismic_df, target_depths)
    merged_df = pd.merge(merged_df, seismic_aligned[['Depth', 'Vp', 'Vs']], on='Depth', how='left')
    
    # 3. 處理地質文字並解碼
    litho_df = parse_json_strata_to_dataframe(strata_list)
    merged_df['Lithology_Desc'] = merged_df['Depth'].apply(
        lambda d: map_interval_data(d, litho_df, '上限深度', '下限深度', '岩石或土壤性質描述')
    ).fillna('Unknown')
    
    merged_df['Lithology_ID'], _ = zip(
        *merged_df['Lithology_Desc'].apply(advanced_geology_encoder)
    )
    
    # 4. 產出最終 (N, 4) 特徵矩陣：[Depth, Vp, Vs, Lithology_ID]
    feature_matrix = merged_df[['Depth', 'Vp', 'Vs', 'Lithology_ID']].to_numpy()
    
    return feature_matrix, merged_df