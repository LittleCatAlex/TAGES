import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from data_fetcher import fetch_tvm_data

def process_velocity(tvm_df: pd.DataFrame, target_depths: np.ndarray) -> pd.DataFrame:
    """
    接收震波速率 DataFrame，同時內插 Vp 與 Vs 到目標深度網格上，回傳 DataFrame
    """
    tvm_df.columns = tvm_df.columns.str.strip().str.capitalize()
    
    # 準備用來存放結果的 DataFrame
    result_df = pd.DataFrame({'Depth': target_depths})

    # 清洗與排序，避免除以零錯誤
    clean_df = tvm_df.groupby('Depth', as_index=False).mean().sort_values(by='Depth')

    # 處理 Vp
    if 'Vp' in clean_df.columns:
        interp_vp = interp1d(clean_df['Depth'], clean_df['Vp'], kind='linear', fill_value="extrapolate")
        result_df['Vp'] = interp_vp(target_depths)
    else:
        result_df['Vp'] = 2500.0

    # 處理 Vs (新增)
    if 'Vs' in clean_df.columns:
        interp_vs = interp1d(clean_df['Depth'], clean_df['Vs'], kind='linear', fill_value="extrapolate")
        result_df['Vs'] = interp_vs(target_depths)
    else:
        result_df['Vs'] = 1500.0 # 給予 Vs 一個合理的預設背景值

    return result_df