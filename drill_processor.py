# drill_processor.py
import pandas as pd
import numpy as np
from config import LITHOLOGY_DICT, NLP_PRIORITY

def map_interval_data(target_depth, interval_df, top_col, bottom_col, target_col):
    """將目標深度映射到對應的區間數值"""
    mask = (interval_df[top_col] <= target_depth) & (interval_df[bottom_col] > target_depth)
    match = interval_df[mask]
    if not match.empty:
        return match.iloc[0][target_col]
    return np.nan

def advanced_geology_encoder(description):
    """地質文字解碼器"""
    if pd.isna(description) or str(description).strip() == "":
        return 0, 0
    desc = str(description)
    
    # 構造特徵 (維持原樣)
    if any(k in desc for k in ["斷層", "斷層泥", "剪裂", "擦痕", "剪切", "破碎帶"]): struct_id = 5
    elif any(k in desc for k in ["破碎", "角礫", "碎裂", "嚴重風化", "鬆散"]): struct_id = 4
    elif any(k in desc for k in ["節理", "裂隙", "裂理", "劈理"]): struct_id = 3
    elif any(k in desc for k in ["層理", "葉理", "紋理", "交錯層", "沉積紋"]): struct_id = 2
    elif any(k in desc for k in ["完整", "緻密", "堅硬", "塊狀"]): struct_id = 1
    else: struct_id = 0
        
    # 岩相分類 (動態讀取字典與優先級！)
    litho_id = 0
    for pid in NLP_PRIORITY:
        if any(k in desc for k in LITHOLOGY_DICT[pid]["keywords"]):
            litho_id = pid
            break
            
    return litho_id, struct_id

def parse_json_strata_to_dataframe(strata_list):
    """將 JSON 的 strata 清單轉換為標準的區間 DataFrame (含上限深度與下限深度)"""
    records = []
    top_depth = 0.0
    for layer in strata_list:
        bottom_depth = float(layer['depth_m'])
        desc = layer['description']
        records.append({
            '上限深度': top_depth,
            '下限深度': bottom_depth,
            '岩石或土壤性質描述': desc
        })
        top_depth = bottom_depth
    return pd.DataFrame(records)

def process_drill_records(merged_df: pd.DataFrame, strata_list: list) -> pd.DataFrame:
    """
    接收統一網格與 JSON 格式的 strata 清單，執行區間映射與文字編碼
    """
    # 1. 將 JSON strata 轉為區間 DataFrame
    litho_df = parse_json_strata_to_dataframe(strata_list)

    # 2. 映射岩性描述
    merged_df['Lithology_Desc'] = merged_df['Depth'].apply(
        lambda d: map_interval_data(d, litho_df, '上限深度', '下限深度', '岩石或土壤性質描述')
    ).fillna('Unknown')

    # 3. 因為 JSON 來源無獨立 RQD，預設填 0.0
    merged_df['RQD'] = 0.0

    # 4. 執行 NLP 特徵解碼
    merged_df['Lithology_ID'], merged_df['Structure_ID'] = zip(
        *merged_df['Lithology_Desc'].apply(advanced_geology_encoder)
    )
    
    return merged_df