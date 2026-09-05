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
    attach_seismic: bool = typer.Option(True, help="是否自動抓取已處理的 TVM 震波資料進行合併")
):
    """
    ⛏️ 讀取 JSON 柱狀圖紀錄、執行 NLP 文字解碼、掛載震波並輸出 AI 訓練矩陣
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(json_path):
        typer.secho(f"❌ 找不到 JSON 鑽探檔案 [{json_path}]！", fg=typer.colors.RED)
        return

    typer.secho(f"\n📂 正在載入與解析 JSON 鑽探資料 [{json_path}]...", fg=typer.colors.CYAN)
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 檢查是否需要合併震波資料
    tvm_processed_path = os.path.join(output_dir, "TVM_Processed.csv")
    tvm_df = pd.DataFrame()
    if attach_seismic:
        if os.path.exists(tvm_processed_path):
            tvm_df = pd.read_csv(tvm_processed_path)
            typer.secho("🔗 已成功掛載預先處理好的震波速率模型", fg=typer.colors.BLUE)
        else:
            typer.secho("⚠️ 找不到已處理的 TVM 檔案，將略過震波合併 (建議先執行 seismic 指令)", fg=typer.colors.YELLOW)

    success_count = 0
    for file_key, info in raw_data.items():
        well_name = info['borehole_info']['borehole_id']
        strata_list = info['strata']
        
        typer.secho(f"\n⏳ 正在處理鑽孔：{well_name} ({file_key})", fg=typer.colors.YELLOW)
        try:
            # 決定最大深度 (取 strata 裡最後一層的深度，若無則預設 10m)
            max_depth = float(strata_list[-1]['depth_m']) if strata_list else 10.0
            if pd.isna(max_depth) or max_depth <= 0.0:
                max_depth = 10.0
                
            target_depths = np.arange(0.0, max_depth + step, step)
            merged_df = pd.DataFrame({'Depth': target_depths, 'Well_Name': well_name})

            # 呼叫更新後的鑽探處理模組 (傳入 strata 清單)
            merged_df = process_drill_records(merged_df, strata_list)

            # 合併震波資料 (如果有的話)
            if not tvm_df.empty:
                merged_df['Depth'] = merged_df['Depth'].round(2)
                tvm_df['Depth'] = tvm_df['Depth'].round(2)
                
                merged_df = pd.merge(merged_df, tvm_df[['Depth', 'Vp', 'Vs']], on='Depth', how='left')
                merged_df['Vp'] = merged_df['Vp'].fillna(2500.0)
                merged_df['Vs'] = merged_df['Vs'].fillna(1500.0)
                feature_cols = ['Depth', 'Vp', 'Vs', 'RQD', 'Lithology_ID', 'Structure_ID']
            else:
                feature_cols = ['Depth', 'RQD', 'Lithology_ID', 'Structure_ID']

            safe_name = file_key.replace('.png', '').replace(' ', '_')
            csv_out = os.path.join(output_dir, f"{safe_name}_Processed.csv")
            npy_out = os.path.join(output_dir, f"{safe_name}_Features.npy")
            
            merged_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
            np.save(npy_out, merged_df[feature_cols].to_numpy())
            
            typer.secho(f"  └─ ✅ {file_key} 處理完成", fg=typer.colors.GREEN)
            success_count += 1
            
        except Exception as e:
            typer.secho(f"  └─ ❌ {file_key} 處理失敗: {e}", fg=typer.colors.RED)

    typer.secho(f"\n🎉 批次處理結束！共成功處理 {success_count} 筆鑽探資料。", fg=typer.colors.MAGENTA, bold=True)

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
    typer.secho("  👉 export-rqd: 將Excel中的RQD分頁匯出成.csv檔")
    typer.secho("  👉 seismic   : 單獨處理震波速率模型")
    typer.secho("  👉 drilling  : 批次處理鑽探紀錄 (自動掛載震波資料與特徵對齊)")
    typer.secho("  👉 fetch     : 從中研院下載震測模型數據(ex:fetch --lat1 24.0 --lon1 121.0 --lat2 24.5 --lon2 121.5)")
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