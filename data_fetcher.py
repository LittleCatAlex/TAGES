import requests
import urllib.parse
import zipfile
import io
import pandas as pd
import urllib3
import os
import time

# 消除 verify=False 產生的煩人警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_tvm_data(lat1: float, lon1: float, lat2: float, lon2: float, depth: int, output_dir: str) -> bool:
    """
    從中研院 API 獲取震波速率剖面資料
    """
    api_url = "https://tecdc.earth.sinica.edu.tw/TWtomo/php/runProfvelorPlot.php"
    base_url = "https://tecdc.earth.sinica.edu.tw/TWtomo/"
    
    # 組合經緯度字串 (確保沒有多餘的空白)
    secret_str = f"{lat1},{lon1},{lat2},{lon2}"
    
    # 強力瀏覽器偽裝 (欺騙伺服器我們是正常的 Google Chrome 瀏覽器)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://tecdc.earth.sinica.edu.tw/TWtomo/",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    
    payload = {
        "secret": secret_str,
        "eq_width": "5",
        "model": "LEE2025", # 注意：如果還是失敗，可以嘗試把這裡改成舊版如 "LEE2015"
        "depth": str(depth),
        "grid": "2",
        "cpt": "0",
        "seisoption": "1", 
        "cptopt": "seis",
        "cptinvert": "0"
    }

    session = requests.Session()
    try:
        # 加入 headers 送出請求
        response = session.post(api_url, data=payload, headers=headers, timeout=60, verify=False)
        response.raise_for_status()
        
        # 【關鍵除錯區】嘗試解析 JSON，並攔截真實錯誤訊息
        try:
            result_data = response.json()
        except ValueError:
            error_msg = response.text[:500]
            raise ValueError(f"伺服器回傳了非預期的格式。\n伺服器回應內容：\n{error_msg}")
            
        # 彈性支援新舊版 API 的 JSON 結構
        if isinstance(result_data, dict):
            if "dir" in result_data:
                dir_id = str(result_data["dir"])
            elif "1" in result_data:  # 處理新版 API 格式
                dir_id = str(result_data["1"])
            else:
                raise ValueError(f"找不到目錄 ID。\n完整回傳內容：{result_data}")
        else:
            raise ValueError(f"回傳格式錯誤。\n完整回傳內容：{result_data}")
            
        # 組合 ZIP 檔案路徑
        zip_path = f"output/{dir_id}/slice.zip"
        full_zip_url = urllib.parse.urljoin(base_url, zip_path)
        
        # ⚠️ 關鍵防呆：因為伺服器在後台跑運算需要一點時間，我們強制讓程式稍等 3 秒，避免伺服器還沒壓縮完我們就去載
        time.sleep(3) 
        
        # 下載 ZIP (同樣帶上 headers)
        zip_response = session.get(full_zip_url, headers=headers, verify=False)
        
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
            file_names = z.namelist()
            data_files = [f for f in file_names if f.endswith(('.txt', '.out', '.dat', '.csv'))]
            
            if data_files:
                target_file = data_files[0]
                with z.open(target_file) as f:
                    # 【關鍵修正】加上 header=None，阻止 Pandas 把第一筆資料吃掉
                    df = pd.read_csv(f, sep=r'\s+', header=None)
                    
                    # 【手動貼標籤】根據中研院格式，指定欄位意義
                    # 第 0 欄是深度, 第 2 欄是 Vp, 第 4 欄是 Vs, 第 8 欄是經度, 第 9 欄是緯度
                    df.rename(columns={0: 'Depth', 2: 'Vp', 4: 'Vs', 8: 'Lon', 9: 'Lat'}, inplace=True)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    output_csv = os.path.join(output_dir, "TVM_VerticalProfile_Output.csv")
                    df.to_csv(output_csv, index=False)
                    return True
            else:
                raise FileNotFoundError("壓縮檔內找不到任何數據檔")
                
    except Exception as e:
        raise Exception(f"下載或解析失敗:\n{str(e)}")