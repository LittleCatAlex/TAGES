# TAGES (Taiwan AI Geological Exploration System)

## 專案簡介

TAGES，用來做科展的。

---

## 安裝與執行

* **下載與建置環境**：執行 `install_env.bat` 批次檔，自動安裝相依套件與設定運行環境。
* **啟動系統**：執行 `start.bat` 批次檔，即可開始運行 TAGES 系統。

---

## 指令指南與參數範例

進入系統後，支援以下指令。您可以直接輸入指令，或加上自訂參數來改變預設行為：

| 指令                     | 功能說明                                                                        | 參數打法範例 |
| :----------------------- | :------------------------------------------------------------------------------ | :--- |
| **`repl`**               | 啟動 TAGES 互動式指令介面。                                                     | `repl` |
| **`fetch`**              | 從中研院地球科學網 (TEC API) 自動下載並轉檔最新震波模型資料。                   | `fetch --lat1 23.50 --lon1 120.18 --lat2 24.50 --lon2 121.50 --depth 60` |
| **`seismic`**            | 獨立處理震波速率模型，執行深度網格內插並輸出特徵矩陣。                          | `seismic --step 0.5 --max-depth 100.0` |
| **`export-rqd`**         | 將 Excel 鑽探檔案中的「岩石RQD值」分頁，批次匯出成乾淨的 CSV 檔。               | `export-rqd --drill-dir data_input/drilling` |
| **`drilling`**           | 批次處理 JSON 鑽探紀錄，自動執行文字解碼、掛載震波資料並進行特徵對齊。          | `drilling --step 0.5 --attach-seismic` |
| **`train`**              | 讀取所有特徵矩陣，訓練「隨機森林 (Random Forest)」AI 模型並自動存檔與產出報告。 | `train --force` (或使用簡寫 `train -f`) |
| **`predict`**            | 啟動互動式 AI 預測終端，手動輸入深度、Vp、Vs 即可即時預測岩石種類。             | `predict` (進入後依照提示輸入即可) |
| **`ai-profile`**         | 畫出預測岩層剖面圖（支援兩點連線或預設緯度剖面）                                | 預設：`ai-profile --max-depth 0.1` <br> 連線：`ai-profile --lat1 23.5 --lon1 120.1 --lat2 24.0 --lon2 120.8 --max-depth 0.1` |
| **`clean`**              | 一鍵清理並刪除`data_output` 內的產出檔案（內建防呆確認機制）。                  | `clean --output-dir data_output` |

> **💡 參數查詢提示**：若忘記參數名稱，隨時可以在任何指令後方加上 `--help`（例如：`fetch --help`），系統會自動列出完整的參數說明與預設值。

---

## ongoing

接下來可能要做的：

* **訓練其他的算法**：
* **優化程式**：