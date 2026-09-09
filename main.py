# TAGES (Taiwan AI Geological Exploration System)
import typer
import click
import shlex
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import LITHOLOGY_DICT
from velocity_processor import process_velocity
from drill_processor import process_drill_records
from data_fetcher import fetch_tvm_data

app = typer.Typer(help="台灣地質資料處理與 AI 訓練特徵萃取工具")


# =====================================================================
# 指令 0：從雲端獲取最新震波資料
# =====================================================================
@app.command(name="fetch")
def fetch_cloud_data(
    lat1: float = typer.Option(23.50, help="起點緯度"),
    lon1: float = typer.Option(120.18, help="起點經度"),
    lat2: float = typer.Option(23.50, help="終點緯度"),
    lon2: float = typer.Option(120.78, help="終點經度"),
    depth: int = typer.Option(60, help="剖面深度(公里)"),
    output_dir: str = typer.Option("data_input/seismic", help="存檔資料夾")
):
    """
    🌐 從中研院地球科學網自動下載並轉檔最新震波模型
    """
    typer.secho(f"\n📡 正在連線至中研院 TEC API...", fg=typer.colors.CYAN)
    typer.secho(f"   座標區間: ({lat1}, {lon1}) 到 ({lat2}, {lon2})", fg=typer.colors.CYAN)
    
    try:
        success = fetch_tvm_data(lat1, lon1, lat2, lon2, depth, output_dir)
        if success:
            typer.secho(f"✅ 下載與轉檔成功！檔案已存至 {output_dir}/TVM_VerticalProfile_Output.csv", fg=typer.colors.GREEN)
            typer.secho(f"👉 下一步建議：輸入 'seismic' 指令進行內插處理。", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"❌ 發生錯誤: {e}", fg=typer.colors.RED)


# =====================================================================
# 指令 1：獨立處理震波速率資料
# =====================================================================
@app.command(name="seismic")
def process_seismic(
    seis_dir: str = typer.Option("data_input/seismic", help="震波速率資料資料夾"),
    output_dir: str = typer.Option("data_output", help="輸出資料夾"),
    step: float = typer.Option(0.5, help="目標深度網格解析度(公尺)"),
    max_depth: float = typer.Option(100.0, help="預設最大深度(公尺)")
):
    """
    🌊 處理震波速率模型 (TVM)，執行深度網格內插並輸出特徵矩陣
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tvm_path = os.path.join(seis_dir, "TVM_VerticalProfile_Output.csv")
    
    typer.secho(f"\n🌊 開始處理震波資料: {tvm_path}", fg=typer.colors.CYAN)
    
    if not os.path.exists(tvm_path):
        typer.secho("❌ 找不到 TVM 檔案，請確認檔案名稱與路徑！", fg=typer.colors.RED)
        return

    try:
        tvm_df = pd.read_csv(tvm_path)
        actual_max = tvm_df['Depth'].max() if 'Depth' in tvm_df.columns else max_depth
        target_depths = np.arange(0.0, actual_max + step, step)
        
        merged_df = process_velocity(tvm_df, target_depths)
        
        csv_out = os.path.join(output_dir, "TVM_Processed.csv")
        npy_out = os.path.join(output_dir, "TVM_Features.npy")
        
        merged_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
        np.save(npy_out, merged_df[['Depth', 'Vp', 'Vs']].to_numpy())
        
        typer.secho(f"✅ 震波資料處理完成！已存至 {output_dir}/TVM_Processed.csv", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ 震波資料處理失敗: {e}", fg=typer.colors.RED)


# =====================================================================
# 指令 2：單獨將 RQD 分頁轉換為 CSV
# =====================================================================
@app.command(name="export-rqd")
def export_rqd_to_csv(
    drill_dir: str = typer.Option("data_input/drilling", help="鑽探原始 Excel 資料夾"),
    output_dir: str = typer.Option("data_output/rqd_csv", help="RQD CSV 專屬輸出資料夾")
):
    """
    📊 獨立將 Excel 檔案中的「岩石RQD值」分頁匯出為標準 .csv 格式
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    typer.secho(f"\n📊 開始批次擷取並匯出 RQD 分頁...", fg=typer.colors.CYAN)
    
    try:
        drill_files = [f for f in os.listdir(drill_dir) if f.endswith(('.xls', '.xlsx')) and not f.startswith('~')]
    except FileNotFoundError:
        typer.secho("❌ 找不到鑽探資料夾，請確認路徑！", fg=typer.colors.RED)
        return
    
    if not drill_files:
        typer.secho("❌ 找不到任何 Excel 鑽探檔案！", fg=typer.colors.RED)
        return

    success_count = 0
    for file_name in drill_files:
        well_name = file_name.split('.xls')[0]
        try:
            file_path = os.path.join(drill_dir, file_name)
            rqd_df = pd.read_excel(file_path, sheet_name='岩石RQD值')
            
            csv_out = os.path.join(output_dir, f"{well_name}_RQD.csv")
            rqd_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
            
            typer.secho(f"  └─ ✅ {well_name} -> {well_name}_RQD.csv 轉換成功", fg=typer.colors.GREEN)
            success_count += 1
        except Exception as e:
            typer.secho(f"  └─ ❌ {well_name} 轉換 RQD 失敗: {e}", fg=typer.colors.RED)
            
    typer.secho(f"\n🎉 RQD 格式轉換完成！共成功導出 {success_count} 個 CSV 檔，已存至：{output_dir}/", fg=typer.colors.MAGENTA, bold=True)


# =====================================================================
# 指令 3：批次處理鑽探資料與震波對齊 (物理重構版)
# =====================================================================
@app.command(name="drilling")
def process_drilling(
    json_path: str = typer.Option("data_input/drilling/borehole_final_results.json", help="柱狀圖 JSON 檔案路徑"),
    output_dir: str = typer.Option("data_output", help="輸出資料夾"),
    step: float = typer.Option(0.5, help="目標深度網格解析度(公尺)"),
    attach_seismic: bool = typer.Option(True, help="是否自動掛載真實震波資料")
):
    """
    ⛏️ 以真實經緯度與物理梯度進行鑽探與震波的特徵對齊
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(json_path):
        typer.secho(f"❌ 找不到 JSON 檔案 [{json_path}]", fg=typer.colors.RED)
        return

    typer.secho(f"\n📂 正在載入 JSON 鑽探資料與原始震波模型...", fg=typer.colors.CYAN)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 🌟 讀取「原始包含經緯度」的 TVM CSV，而不是被閹割過的 Processed 版本
    tvm_df = pd.DataFrame()
    tvm_raw_path = "data_input/seismic/TVM_VerticalProfile_Output.csv"
    if attach_seismic and os.path.exists(tvm_raw_path):
        tvm_df = pd.read_csv(tvm_raw_path)
        typer.secho("🔗 成功找到實測 3D 震波模型！啟動經緯度對齊與物理內插機制。", fg=typer.colors.BLUE)

    success_count = 0
    for file_key, info in raw_data.items():
        well_name = info['borehole_info']['borehole_id']
        strata_list = info['strata']
        
        # 取得鑽孔經緯度
        well_lon = float(info['borehole_info'].get('lon', 120.0))
        well_lat = float(info['borehole_info'].get('lat', 23.5))
        
        try:
            max_depth = float(strata_list[-1]['depth_m']) if strata_list else 10.0
            if pd.isna(max_depth) or max_depth <= 0.0: max_depth = 10.0
                
            target_depths = np.arange(0.0, max_depth + step, step)
            merged_df = pd.DataFrame({'Depth': target_depths, 'Well_Name': well_name})
            merged_df = process_drill_records(merged_df, strata_list)

            # 🌟 步驟一：真實經緯度搜尋 + 深度線性內插
            if not tvm_df.empty and 'Lon' in tvm_df.columns and 'Lat' in tvm_df.columns:
                distances = np.sqrt((tvm_df['Lon'] - well_lon)**2 + (tvm_df['Lat'] - well_lat)**2)
                closest_idx = distances.idxmin()
                
                closest_lon = tvm_df.loc[closest_idx, 'Lon']
                closest_lat = tvm_df.loc[closest_idx, 'Lat']
                
                local_tvm = tvm_df[(tvm_df['Lon'] == closest_lon) & (tvm_df['Lat'] == closest_lat)].copy()
                local_tvm = local_tvm.sort_values(by='Depth')

                merged_df['Vp'] = np.interp(merged_df['Depth'], local_tvm['Depth'], local_tvm['Vp'])
                merged_df['Vs'] = np.interp(merged_df['Depth'], local_tvm['Depth'], local_tvm['Vs'])
            else:
                merged_df['Vp'] = np.nan
                merged_df['Vs'] = np.nan

            # 🌟 步驟二：加入自然壓實梯度的經驗值填補
            empirical_vel = {k: (v["vp"], v["vs"]) for k, v in LITHOLOGY_DICT.items()}
            
            for litho_id, (vp_emp, vs_emp) in empirical_vel.items():
                mask = (merged_df['Lithology_ID'] == litho_id) & (merged_df['Vp'].isna())
                count = mask.sum()
                if count > 0:
                    depths_masked = merged_df.loc[mask, 'Depth'].values
                    # 加入深度梯度 (每公尺 Vp 加 5, Vs 加 2.5) 與微小雜訊，模擬真實物理現象
                    merged_df.loc[mask, 'Vp'] = vp_emp + (depths_masked * 5.0) + np.random.normal(0, vp_emp * 0.02, count)
                    merged_df.loc[mask, 'Vs'] = vs_emp + (depths_masked * 2.5) + np.random.normal(0, vs_emp * 0.02, count)

            merged_df['Vp'] = merged_df['Vp'].fillna(1500.0)
            merged_df['Vs'] = merged_df['Vs'].fillna(500.0)

            # 輸出
            feature_cols = ['Depth', 'Vp', 'Vs', 'RQD', 'Lithology_ID', 'Structure_ID']
            safe_name = file_key.replace('.png', '').replace(' ', '_')
            merged_df.to_csv(os.path.join(output_dir, f"{safe_name}_Processed.csv"), index=False, encoding='utf-8-sig')
            np.save(os.path.join(output_dir, f"{safe_name}_Features.npy"), merged_df[feature_cols].to_numpy())
            success_count += 1
            
        except Exception as e:
            typer.secho(f"  └─ ❌ {file_key} 處理失敗: {e}", fg=typer.colors.RED)

    typer.secho(f"\n🎉 處理結束！成功產出 {success_count} 筆訓練資料。", fg=typer.colors.MAGENTA, bold=True)

# =====================================================================
# 指令 4：一鍵清除輸出資料夾 (帶防呆)
# =====================================================================
@app.command(name="clean")
def clean_output(
    output_dir: str = typer.Option("data_output", help="要清除的輸出資料夾")
):
    """
    🧹 清除 data_output 資料夾內的所有產出檔案與子資料夾
    """
    if not os.path.exists(output_dir):
        typer.secho(f"ℹ️ 資料夾 [{output_dir}] 本來就是空的，無需清除。", fg=typer.colors.CYAN)
        return

    confirm = typer.confirm(f"⚠️ 確定要清空 [{output_dir}] 資料夾內的所有檔案嗎？(此作業無法復原)", default=False)
    if not confirm:
        typer.secho("❌ 已取消清除作業。", fg=typer.colors.YELLOW)
        return

    try:
        typer.secho(f"\n🧹 正在清理 [{output_dir}] 內的所有內容...", fg=typer.colors.CYAN)
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        typer.secho(f"✅ 清理完成！[{output_dir}] 現在是一片淨土了。", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ 清除失敗，錯誤訊息: {e}", fg=typer.colors.RED)

# =====================================================================
# 指令 5：訓練 AI 模型 (包含深度特徵)
# =====================================================================
@app.command(name="train")
def train_model(
    data_dir: str = typer.Option("data_output", help="特徵矩陣所在資料夾"),
    model_dir: str = typer.Option("models", help="模型存檔資料夾"),
    force: bool = typer.Option(False, "--force", "-f", help="強制重新訓練，不載入舊模型")
):
    """
    讀取特徵矩陣 (Depth, Vp, Vs)，訓練 AI 模型
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tages_rf_model.pkl")
    
    all_data = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith("_Features.npy") and not file.startswith("TVM"):
                all_data.append(np.load(os.path.join(data_dir, file)))
                
    if not all_data:
        typer.secho("❌ 找不到特徵矩陣！請先執行 drilling。", fg=typer.colors.RED)
        return
        
    dataset = np.vstack(all_data)
    cols = dataset.shape[1]
    
    # 🌟 關鍵修正：將 0:3 納入特徵 (包含 Depth, Vp, Vs)
    if cols >= 5:
        X, y = dataset[:, 0:3], dataset[:, 4]
    else:
        X, y = dataset[:, 0:3], dataset[:, 3]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = None
    if not force and os.path.exists(model_path):
        if typer.confirm(f"🔄 發現舊模型 ({model_path})，要直接載入嗎？(選 N 則重新訓練)", default=True):
            model = joblib.load(model_path)
            typer.secho("✅ 舊模型載入成功！", fg=typer.colors.GREEN)
            
    if model is None:
        typer.secho("⏳ AI 正在學習地層壓實與震波梯度...", fg=typer.colors.YELLOW)
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    typer.secho(f"\n📊 總體準確率 (Accuracy): {accuracy * 100:.2f}%", fg=typer.colors.CYAN, bold=True)
    joblib.dump(model, model_path)
    typer.secho(f"💾 模型已儲存至 {model_dir} 資料夾！", fg=typer.colors.GREEN)

# =====================================================================
# 指令 6：啟動 AI 預測終端
# =====================================================================
@app.command(name="predict")
def interactive_predict():
    """
    啟動地質預測終端 (支援深度特徵)
    """
    model_path = os.path.join("models", "tages_rf_model.pkl")
    if not os.path.exists(model_path):
        typer.secho("❌ 找不到訓練好的模型！請先執行 train 指令。", fg=typer.colors.RED)
        return

    model = joblib.load(model_path)
    mapping = {k: v["name"] for k, v in LITHOLOGY_DICT.items()}
    
    typer.secho("\n✅ 模型載入完成！進入互動預測模式 (輸入 q 離開)", fg=typer.colors.GREEN, bold=True)
    
    while True:
        try:
            print("-" * 45)
            depth_input = input("📍 請輸入深度 (公尺): ").strip()
            if depth_input.lower() in ['q', 'quit', 'exit']: break
            
            depth = float(depth_input)
            vp = float(input("🌊 請輸入 P波速率 (Vp, m/s): ").strip())
            vs = float(input("🌊 請輸入 S波速率 (Vs, m/s): ").strip())

            # 🌟 關鍵修正：將 depth 一起送入模型進行預測
            features = np.array([[depth, vp, vs]])
            prediction_id = model.predict(features)[0]
            rock_name = mapping.get(int(prediction_id), "未定義")

            typer.secho(f"\n✨ 預測結果 👉 【 {rock_name} 】", fg=typer.colors.MAGENTA, bold=True)
            
        except ValueError:
            typer.secho("⚠️ 格式錯誤！請確認輸入的是純數字。", fg=typer.colors.RED)
        except Exception as e:
            typer.secho(f"⚠️ 發生錯誤: {e}", fg=typer.colors.RED)
        except KeyboardInterrupt:
            break
            
    typer.secho("\n👋 已離開預測終端。", fg=typer.colors.YELLOW)

# =====================================================================
# 指令 10：繪製 AI 預測的高解析地質剖面圖
# =====================================================================
@app.command(name="ai-profile")
def plot_ai_profile_cmd(
    csv_path: str = typer.Option("data_input/seismic/TVM_VerticalProfile_Output.csv", help="原始震波 CSV 路徑"),
    model_path: str = typer.Option("models/tages_rf_model.pkl", help="訓練好的 AI 模型路徑"),
    max_depth: float = typer.Option(0.1, help="顯示的最大深度 (公里)") 
):
    """
    🤖 繪製 2D 地質剖面 (自動切片處理)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata

    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        typer.secho("❌ 找不到震波檔案或 AI 模型！", fg=typer.colors.RED)
        return

    typer.secho(f"⏳ 正在進行 2D 切片與網格內插...", fg=typer.colors.CYAN)
    
    df = pd.read_csv(csv_path)
    
    # 🌟 關鍵修正：只保留中間緯度附近的一刀切片，避免 3D 資料擠壓
    if 'Lat' in df.columns:
        target_lat = df['Lat'].mean()
        tolerance = 0.05
        df = df[(df['Lat'] >= target_lat - tolerance) & (df['Lat'] <= target_lat + tolerance)]
        typer.secho(f"📍 剖面緯度鎖定於: {target_lat:.3f} ± {tolerance}", fg=typer.colors.YELLOW)

    lon = df['Lon'].values
    depth_km = df['1'].values 
    vp_kms = df['Vp'].values
    vs_kms = df['Vs'].values

    # 地表邊界條件
    unique_lons = np.unique(lon)
    surface_depth = np.zeros_like(unique_lons)
    surface_vp = np.full_like(unique_lons, 1.2)
    surface_vs = np.full_like(unique_lons, 0.4)

    aug_lon = np.concatenate([lon, unique_lons])
    aug_depth = np.concatenate([depth_km, surface_depth])
    aug_vp = np.concatenate([vp_kms, surface_vp])
    aug_vs = np.concatenate([vs_kms, surface_vs])

    x_min, x_max = lon.min(), lon.max()
    grid_x, grid_y = np.mgrid[x_min:x_max:500j, 0:max_depth:500j]

    grid_vp = griddata((aug_lon, aug_depth), aug_vp, (grid_x, grid_y), method='linear')
    grid_vs = griddata((aug_lon, aug_depth), aug_vs, (grid_x, grid_y), method='linear')
    
    grid_vp = np.where(np.isnan(grid_vp), griddata((aug_lon, aug_depth), aug_vp, (grid_x, grid_y), method='nearest'), grid_vp)
    grid_vs = np.where(np.isnan(grid_vs), griddata((aug_lon, aug_depth), aug_vs, (grid_x, grid_y), method='nearest'), grid_vs)

    # 將畫圖用的公里(km)轉成模型吃的公尺(m)
    depth_m = grid_y.flatten() * 1000
    vp_ms = grid_vp.flatten() * 1000
    vs_ms = grid_vs.flatten() * 1000

    # 🌟 關鍵修正：將深度一起餵入模型
    model = joblib.load(model_path)
    features = np.column_stack((depth_m, vp_ms, vs_ms))
    predictions = model.predict(features)
    grid_litho = predictions.reshape(grid_x.shape)

    colors = [LITHOLOGY_DICT[i]["color"] for i in range(len(LITHOLOGY_DICT))]
    labels = [LITHOLOGY_DICT[i]["name"] for i in range(len(LITHOLOGY_DICT))]
    
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(LITHOLOGY_DICT) + 0.5, 1) 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(grid_x, grid_y, grid_litho, cmap=cmap, norm=norm, shading='auto')

    ax.invert_yaxis()
    ax.set_xlabel('Longitude (Degrees)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title(f'TAGES AI Geological Profile (Lat: {target_lat:.2f})', fontsize=16, fontweight='bold')
    
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', title="Lithology")
    
    plt.tight_layout()
    output_png = "data_output/AI_Predicted_Physics_Profile.png"
    plt.savefig(output_png, dpi=300)
    typer.secho(f"✅ 剖面圖渲染完成！已儲存至：{output_png}", fg=typer.colors.GREEN, bold=True)
    plt.show()

# =====================================================================
# CLI 指令：自製互動式 REPL 介面
# =====================================================================
@app.command()
def repl():
    """🚀 啟動REPL介面"""
    typer.secho("=====================================================", fg=typer.colors.MAGENTA)
    typer.secho("   _____  _    ____  _____ ____  \n  |_   _|/ \\  / ___|| ____/ ___| \n  / | | / _ \\| |  _ |  _| \\___ \\ \n  / | |/ ___ \\ |_| || |___ ___) |\n  / |_/_/   \\_\\____||_____|____/ \n  //   //   // //// ///// ///// \n", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("=====================================================", fg=typer.colors.MAGENTA)
    typer.secho("Welcome to TAGES (Taiwan AI Geological Exploration System)\n",  fg=typer.colors.YELLOW)
    typer.secho("可用指令：", fg=typer.colors.CYAN)
    typer.secho("  👉 train     : 訓練隨機森林模型")
    typer.secho("  👉 predict   : 啟動預測終端")
    typer.secho("  👉 export-rqd: 將Excel中的RQD分頁匯出成.csv檔")
    typer.secho("  👉 seismic   : 單獨處理震波速率模型")
    typer.secho("  👉 drilling  : 批次處理鑽探紀錄 (自動掛載震波資料與特徵對齊)")
    typer.secho("  👉 fetch     : 從中研院下載震測模型數據")
    typer.secho("  👉 ai-profile: 畫出滿版預測岩層剖面圖")
    typer.secho("  👉 clean     : 刪除並清空data_output資料夾內的所有產出")
    typer.secho("  👉 exit      : 離開系統\n")
    while True:
        try:
            cmd_str = input("TAGES > ").strip()
            if cmd_str.lower() in ['exit', 'quit']: break
            if not cmd_str: continue
            args = shlex.split(cmd_str)
            if args and args[0].lower() == 'help':
                args = ['--help'] if len(args) == 1 else [args[1], '--help']
            try:
                app(args, standalone_mode=False)
            except click.exceptions.UsageError as e: e.show() 
            except click.exceptions.Exit: pass 
            except Exception as e: typer.secho(f"錯誤: {e}", fg=typer.colors.RED)
        except (KeyboardInterrupt, EOFError): break

if __name__ == "__main__":
    app()