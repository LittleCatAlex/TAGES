# train.py
import os
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_all_features(data_dir="data_output"):
    """讀取資料夾內所有的 .npy 檔案，並垂直疊加成單一巨型矩陣"""
    all_data = []
    for file in os.listdir(data_dir):
        # 兼容批次 drilling 產出的 _Features 與單一井的 _Dynamic_Features
        if file.endswith("_Features.npy") and not file.startswith("TVM"):
            filepath = os.path.join(data_dir, file)
            data = np.load(filepath)
            all_data.append(data)
            
    if not all_data:
        raise ValueError("找不到任何 _Features.npy 檔案，請先執行 drilling 或 single-well 指令產生資料！")
        
    return np.vstack(all_data)

def main():
    print("==================================================")
    print("  TAGES AI 模型訓練與存續系統啟動")
    print("==================================================\n")
    
    # 準備存檔資料夾
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tages_rf_model.pkl")
    report_path = os.path.join(model_dir, "training_report.txt")
    
    # 1. 載入資料
    print("📂 正在從 data_output/ 載入所有特徵矩陣...")
    try:
        dataset = load_all_features()
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return
        
    print(f"✅ 成功載入！總資料筆數: {dataset.shape[0]} 筆\n")
    
    # 2. 切割特徵 (X) 與標籤 (y)
    cols = dataset.shape[1]
    
    if cols >= 5:
        # 這是 drilling 產出的 6 欄矩陣: [Depth, Vp, Vs, RQD, Lithology_ID, Structure_ID]
        X = dataset[:, 0:3]
        y = dataset[:, 4]    # Lithology_ID 在索引 4
    else:
        # 這是 single-well 產出的 4 欄矩陣: [Depth, Vp, Vs, Lithology_ID]
        X = dataset[:, 0:3]
        y = dataset[:, 3]    # Lithology_ID 在索引 3
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🧠 準備就緒！訓練卷有 {X_train.shape[0]} 題，期末考卷有 {X_test.shape[0]} 題。\n")
    
    # 3. 斷點存續系統 (Checkpointing)
    model = None
    if os.path.exists(model_path):
        user_input = input(f"🔄 發現已存在的模型 ({model_path})，要直接載入免重新訓練嗎？(y/n): ").strip().lower()
        if user_input == 'y':
            print("⏳ 正在喚醒沉睡中的 AI 模型...")
            model = joblib.load(model_path)
            print("✅ 模型載入成功！")
    
    # 如果選擇 n 或者根本還沒建立過模型，就開始全新訓練
    if model is None:
        print("⏳ 建立新模型，AI 正在拼命學習中 (Random Forest)...")
        # n_jobs=-1 會火力全開使用你電腦 CPU 的所有核心來訓練
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    
    # 4. 模型期末考與評估
    print("\n📊 模型期末考成績單：")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 建立評估報告文字
    report_str = f"模型訓練與評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_str += f"總資料筆數: {dataset.shape[0]} (訓練: {X_train.shape[0]}, 測試: {X_test.shape[0]})\n"
    report_str += f"總體準確率 (Accuracy): {accuracy * 100:.2f}%\n"
    report_str += "-"*55 + "\n"
    report_str += "詳細分類報告 (Classification Report):\n"
    report_str += classification_report(y_test, y_pred, zero_division=0)
    
    print(report_str)

    # 5. 存檔與寫入報告
    joblib.dump(model, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    print(f"💾 模型已成功存檔至: {model_path}")
    print(f"📝 訓練報告已儲存至: {report_path}")

if __name__ == "__main__":
    main()