#main.py
#TAGES (Taiwan AI Geological Exploration System)
import typer
import click
import shlex
import pandas as pd
import numpy as np
import os
import shutil  # 【新增】用來安全刪除子資料夾與檔案
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
# 指令 3：批次處理鑽探資料與震波對齊
# =====================================================================
@app.command(name="drilling")
def process_drilling(
    json_path: str = typer.Option("data_input/drilling/borehole_final_results.json", help="柱狀圖 JSON 檔案路徑"),
    output_dir: str = typer.Option("data_output", help="輸出資料夾"),
    step: float = typer.Option(0.5, help="目標深度網格解析度(公尺)"),
    attach_seismic: bool = typer.Option(True, help="是否自動掛載真實震波資料")
):
    """
    ⛏️ 優先合併真實 API 震波，若淺層無資料則自動使用物理經驗值填補
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(json_path):
        typer.secho(f"❌ 找不到 JSON 檔案 [{json_path}]", fg=typer.colors.RED)
        return

    typer.secho(f"\n📂 正在載入 JSON 鑽探資料與震波模型...", fg=typer.colors.CYAN)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 嘗試讀取已處理的真實震波資料
    tvm_df = pd.DataFrame()
    tvm_processed_path = os.path.join(output_dir, "TVM_Processed.csv")
    if attach_seismic and os.path.exists(tvm_processed_path):
        tvm_df = pd.read_csv(tvm_processed_path)
        typer.secho("🔗 成功找到實測震波模型！(淺層無資料處將啟動物理字典輔助)", fg=typer.colors.BLUE)

    success_count = 0
    for file_key, info in raw_data.items():
        well_name = info['borehole_info']['borehole_id']
        strata_list = info['strata']
        
        try:
            max_depth = float(strata_list[-1]['depth_m']) if strata_list else 10.0
            if pd.isna(max_depth) or max_depth <= 0.0: max_depth = 10.0
                
            target_depths = np.arange(0.0, max_depth + step, step)
            merged_df = pd.DataFrame({'Depth': target_depths, 'Well_Name': well_name})

            # 解碼岩性文字
            merged_df = process_drill_records(merged_df, strata_list)

            # 🌟 步驟一：保留原本做法，優先合併真實震波
            if not tvm_df.empty:
                merged_df['Depth'] = merged_df['Depth'].round(2)
                tvm_df['Depth'] = tvm_df['Depth'].round(2)
                merged_df = pd.merge(merged_df, tvm_df[['Depth', 'Vp', 'Vs']], on='Depth', how='left')
            else:
                merged_df['Vp'] = np.nan
                merged_df['Vs'] = np.nan

            # 🌟 步驟二：沒有資料 (NaN) 的地方，再用錨定經驗值填補
            empirical_vel = {k: (v["vp"], v["vs"]) for k, v in LITHOLOGY_DICT.items()}
            
            for litho_id, (vp_emp, vs_emp) in empirical_vel.items():
                # 關鍵邏輯：只針對「該岩性」且「Vp 沒抓到真實資料 (isna)」的列進行填補
                mask = (merged_df['Lithology_ID'] == litho_id) & (merged_df['Vp'].isna())
                count = mask.sum()
                if count > 0:
                    merged_df.loc[mask, 'Vp'] = np.random.normal(vp_emp, vp_emp * 0.1, count)
                    merged_df.loc[mask, 'Vs'] = np.random.normal(vs_emp, vs_emp * 0.1, count)

            # 最後的防呆機制：確保絕對沒有遺漏的 NaN
            merged_df['Vp'] = merged_df['Vp'].fillna(1500.0)
            merged_df['Vs'] = merged_df['Vs'].fillna(500.0)

            # 輸出特徵檔案
            feature_cols = ['Depth', 'Vp', 'Vs', 'RQD', 'Lithology_ID', 'Structure_ID']
            safe_name = file_key.replace('.png', '').replace(' ', '_')
            merged_df.to_csv(os.path.join(output_dir, f"{safe_name}_Processed.csv"), index=False, encoding='utf-8-sig')
            np.save(os.path.join(output_dir, f"{safe_name}_Features.npy"), merged_df[feature_cols].to_numpy())
            success_count += 1
            
        except Exception as e:
            typer.secho(f"  └─ ❌ {file_key} 處理失敗: {e}", fg=typer.colors.RED)

    typer.secho(f"\n🎉 處理結束！成功產出 {success_count} 筆訓練資料。", fg=typer.colors.MAGENTA, bold=True)

# =====================================================================
# 【全新功能】指令 4：一鍵清除輸出資料夾 (帶防呆)
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

    # 🚨 防呆確認提示
    confirm = typer.confirm(f"⚠️ 確定要清空 [{output_dir}] 資料夾內的所有檔案嗎？(此作業無法復原)", default=False)
    if not confirm:
        typer.secho("❌ 已取消清除作業。", fg=typer.colors.YELLOW)
        return

    try:
        typer.secho(f"\n🧹 正在清理 [{output_dir}] 內的所有內容...", fg=typer.colors.CYAN)
        
        # 遍歷刪除裡面的所有檔案與子資料夾
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 刪除子資料夾 (例如 rqd_csv/)
            else:
                os.remove(item_path)     # 刪除獨立檔案
                
        typer.secho(f"✅ 清理完成！[{output_dir}] 現在是一片淨土了。", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ 清除失敗，錯誤訊息: {e}", fg=typer.colors.RED)

# =====================================================================
# 指令 5：訓練 AI 模型 (Train)
# =====================================================================
@app.command(name="train")
def train_model(
    data_dir: str = typer.Option("data_output", help="特徵矩陣所在資料夾"),
    model_dir: str = typer.Option("models", help="模型存檔資料夾"),
    force: bool = typer.Option(False, "--force", "-f", help="強制重新訓練，不載入舊模型")
):
    """
    讀取所有特徵矩陣，訓練隨機森林 AI 模型並自動存檔
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tages_rf_model.pkl")
    report_path = os.path.join(model_dir, "training_report.txt")
    
    all_data = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith("_Features.npy") and not file.startswith("TVM"):
                all_data.append(np.load(os.path.join(data_dir, file)))
                
    if not all_data:
        typer.secho("❌ 找不到特徵矩陣！請先執行 drilling 或 single-well。", fg=typer.colors.RED)
        return
        
    dataset = np.vstack(all_data)
    cols = dataset.shape[1]
    
    # 自動相容 6欄(drilling) 與 4欄(single-well) 的矩陣
    # 自動相容 6欄(drilling) 與 4欄(single-well) 的矩陣
    if cols >= 5:
        # 【關鍵修改】特徵 (X) 從 1:3 抓取，代表只拿索引 1 (Vp) 和 2 (Vs)，拋棄 0 (Depth)
        X, y = dataset[:, 1:3], dataset[:, 4]
    else:
        # 【關鍵修改】同上
        X, y = dataset[:, 1:3], dataset[:, 3]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = None
    if not force and os.path.exists(model_path):
        if typer.confirm(f"🔄 發現舊模型 ({model_path})，要直接載入嗎？(選 N 則重新訓練)", default=True):
            model = joblib.load(model_path)
            typer.secho("✅ 舊模型載入成功！", fg=typer.colors.GREEN)
            
    if model is None:
        typer.secho("⏳ AI 正在拼命學習中...", fg=typer.colors.YELLOW)
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    typer.secho(f"\n📊 總體準確率 (Accuracy): {accuracy * 100:.2f}%", fg=typer.colors.CYAN, bold=True)
    
    joblib.dump(model, model_path)
    typer.secho(f"💾 模型與報告已儲存至 {model_dir} 資料夾！", fg=typer.colors.GREEN)


# =====================================================================
# 指令 6：啟動 AI 預測終端 (Predict)
# =====================================================================
@app.command(name="predict")
def interactive_predict():
    """
    啟動地質預測終端
    """
    import os
    import joblib
    import numpy as np
    
    model_path = os.path.join("models", "tages_rf_model.pkl")
    if not os.path.exists(model_path):
        typer.secho("❌ 找不到訓練好的模型！請先執行 train 指令。", fg=typer.colors.RED)
        return

    model = joblib.load(model_path)
    
    # 🌟 自動載入全域預測對照表
    mapping = {k: v["name"] for k, v in LITHOLOGY_DICT.items()}
    
    typer.secho("\n✅ 模型載入完成！進入互動預測模式 (輸入 q 離開)", fg=typer.colors.GREEN, bold=True)
    typer.secho("💡 提示：", fg=typer.colors.YELLOW)
    
    while True:
        try:
            print("-" * 45)
            depth_input = input("📍 請輸入深度 (公尺): ").strip()
            if depth_input.lower() in ['q', 'quit', 'exit']: break
            
            depth = float(depth_input)
            vp = float(input("🌊 請輸入 P波速率 (Vp, m/s): ").strip())
            vs = float(input("🌊 請輸入 S波速率 (Vs, m/s): ").strip())

            # 🌟 關鍵修正：AI 模型現在只有 2 個特徵 (Vp, Vs)，所以不要把 depth 丟進去！
            features = np.array([[vp, vs]])
            prediction_id = model.predict(features)[0]
            rock_name = mapping.get(int(prediction_id), "未定義")

            typer.secho(f"\n✨ 預測結果 👉 【 {rock_name} 】 (類別ID: {int(prediction_id)})", fg=typer.colors.MAGENTA, bold=True)
            
        except ValueError as e:
            # 區分是使用者打錯字，還是模型報錯
            if "could not convert" in str(e).lower() or not str(e):
                typer.secho("⚠️ 格式錯誤！請確認輸入的是純數字。", fg=typer.colors.RED)
            else:
                typer.secho(f"⚠️ 模型運算發生錯誤: {e}", fg=typer.colors.RED)
        except KeyboardInterrupt:
            break
            
    typer.secho("\n👋 已離開預測終端。", fg=typer.colors.YELLOW)

# =====================================================================
# 指令 10：繪製 AI 預測的高解析地質剖面圖 (Plot AI Profile)
# =====================================================================
@app.command(name="ai-profile")
def plot_ai_profile_cmd(
    csv_path: str = typer.Option("data_input/seismic/TVM_VerticalProfile_Output.csv", help="原始震波 CSV 路徑"),
    model_path: str = typer.Option("models/tages_rf_model.pkl", help="訓練好的 AI 模型路徑"),
    max_depth: float = typer.Option(2.0, help="顯示的最大深度 (公里)")
):
    """
    🤖 加入地質「壓實梯度」與「地表邊界條件」，進行最合理的 AI 岩相預測
    """
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from scipy.interpolate import griddata

    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        typer.secho("❌ 找不到震波檔案或 AI 模型！", fg=typer.colors.RED)
        return

    typer.secho(f"⏳ 正在進行自然壓實梯度內插...", fg=typer.colors.CYAN)
    
    # 1. 讀取 API 深層震波數據
    df = pd.read_csv(csv_path)
    lon = df['Lon'].values
    depth_km = df['1'].values 
    vp_kms = df['Vp'].values
    vs_kms = df['Vs'].values

    # 🌟【最合理的物理解法：加入地表虛擬測站】
    # 我們在地表 (Depth=0) 建立一整排的基準點，賦予未壓實土壤的標準波速
    unique_lons = np.unique(lon)
    surface_depth = np.zeros_like(unique_lons)
    surface_vp = np.full_like(unique_lons, 1.2)  # 地表 Vp 預設為 1200 m/s (1.2 km/s)
    surface_vs = np.full_like(unique_lons, 0.4)  # 地表 Vs 預設為 400 m/s (0.4 km/s)

    # 將地表基準點與 API 深層數據「合併」
    aug_lon = np.concatenate([lon, unique_lons])
    aug_depth = np.concatenate([depth_km, surface_depth])
    aug_vp = np.concatenate([vp_kms, surface_vp])
    aug_vs = np.concatenate([vs_kms, surface_vs])

    # 2. 建立高解析度繪圖網格
    x_min, x_max = lon.min(), lon.max()
    grid_x, grid_y = np.mgrid[x_min:x_max:500j, 0:max_depth:500j]

    # 3. 🌟【改用 Linear 內插產生自然梯度】
    # linear 會在 0km(1.2) 到 1km(3.8) 之間拉出一條平滑的物理過渡帶
    grid_vp = griddata((aug_lon, aug_depth), aug_vp, (grid_x, grid_y), method='linear')
    grid_vs = griddata((aug_lon, aug_depth), aug_vs, (grid_x, grid_y), method='linear')
    
    # 防呆：填補 linear 算不到的邊界角落
    grid_vp = np.where(np.isnan(grid_vp), griddata((aug_lon, aug_depth), aug_vp, (grid_x, grid_y), method='nearest'), grid_vp)
    grid_vs = np.where(np.isnan(grid_vs), griddata((aug_lon, aug_depth), aug_vs, (grid_x, grid_y), method='nearest'), grid_vs)

    # 4. 單位轉換 (km -> m)
    vp_ms = grid_vp.flatten() * 1000
    vs_ms = grid_vs.flatten() * 1000

    # 5. 載入 AI 模型進行預測 (只看 Vp, Vs)
    model = joblib.load(model_path)
    features = np.column_stack((vp_ms, vs_ms))
    predictions = model.predict(features)
    grid_litho = predictions.reshape(grid_x.shape)

    # 6. 色票與圖例 (全自動對應，未來加到 100 種也不怕報錯！)
    colors = [LITHOLOGY_DICT[i]["color"] for i in range(len(LITHOLOGY_DICT))]
    labels = [LITHOLOGY_DICT[i]["name"] for i in range(len(LITHOLOGY_DICT))]
    
    cmap = mcolors.ListedColormap(colors)
    # 動態計算邊界，有幾種分類就自動切幾格
    bounds = np.arange(-0.5, len(LITHOLOGY_DICT) + 0.5, 1) 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(grid_x, grid_y, grid_litho, cmap=cmap, norm=norm, shading='auto')

    ax.invert_yaxis()
    ax.set_xlabel('Longitude (Degrees)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title(f'TAGES AI Geological Profile (Physics-Anchored, max {max_depth} km)', fontsize=16, fontweight='bold')
    
    labels = ["Unknown", "Topsoil", "Mud/Clay", "Silt", "Sandstone", "Gravel", "Interbedded", "Hard Bedrock"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', title="Lithology")
    
    plt.tight_layout()
    output_png = "data_output/AI_Predicted_Physics_Profile.png"
    plt.savefig(output_png, dpi=300)
    typer.secho(f"✅ 符合物理法則的剖面圖渲染完成！已儲存至：{output_png}", fg=typer.colors.GREEN, bold=True)
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
    typer.secho("  👉 fetch     : 從中研院下載震測模型數據(ex:fetch --lat1 24.0 --lon1 121.0 --lat2 24.5 --lon2 121.5)")
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