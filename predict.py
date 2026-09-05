# predict.py
import joblib
import numpy as np
import os

def get_lithology_name(litho_id):
    """將 AI 輸出的數字 ID 轉換回人類看得懂的地質名稱"""
    mapping = {
        0: "未知 / 其他",
        1: "泥岩 / 頁岩 / 黏土 / 粉土 (細顆粒)",
        2: "砂岩 (中顆粒)",
        3: "礫石 / 卵石 / 巖塊 (粗顆粒)",
        4: "互層"
    }
    return mapping.get(int(litho_id), "未定義")

def main():
    print("==================================================")
    print("  🤖 TAGES AI 智慧地質預測終端")
    print("==================================================\n")

    model_path = os.path.join("models", "tages_rf_model.pkl")

    if not os.path.exists(model_path):
        print("❌ 找不到訓練好的模型！請先執行 python train.py 進行訓練。")
        return

    print("⏳ 載入 AI 模型中...")
    model = joblib.load(model_path)
    print("✅ 模型載入完成！現在你可以自由輸入參數來考驗 AI 了。\n")

    while True:
        try:
            print("-" * 45)
            depth_input = input("📍 請輸入深度 (公尺, 輸入 q 離開): ")
            if depth_input.lower() in ['q', 'quit', 'exit']:
                break
                
            depth = float(depth_input)
            vp = float(input("🌊 請輸入 P波速率 (Vp, m/s): "))
            vs = float(input("🌊 請輸入 S波速率 (Vs, m/s): "))

            # 組裝給 AI 判斷的特徵矩陣 (1筆資料, 3個特徵)
            features = np.array([[depth, vp, vs]])

            # 進行預測
            prediction_id = model.predict(features)[0]
            rock_name = get_lithology_name(prediction_id)

            print("\n✨ AI 預測結果 ✨")
            print(f"👉 物理條件: 深度 {depth}m | Vp {vp} m/s | Vs {vs} m/s")
            print(f"👉 岩相判定: 【 {rock_name} 】 (類別 ID: {int(prediction_id)})\n")

        except ValueError:
            print("⚠️ 格式錯誤！請輸入正確的數字。")
        except KeyboardInterrupt:
            break

    print("\n👋 測驗結束，已離開 AI 預測終端。")

if __name__ == "__main__":
    main()