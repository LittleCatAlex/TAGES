@echo off
:: 設定編碼為UTF-8
chcp 65001 >nul
:: 鎖定工作路徑至批次檔所在目錄
cd /d "%~dp0"

title TAGES系統終端機
color 0A

echo =====================================================
echo  正在啟動 TAGES (Taiwan AI Geological Exploration System)...
echo =====================================================
echo.

:: 全自動環境偵測邏輯
if exist venv\Scripts\activate.bat (
    echo [狀態] 偵測到虛擬環境，啟動中...
    call venv\Scripts\activate.bat
) else (
    echo [狀態] 未偵測到虛擬環境，自動退回使用全域環境執行...
    echo         (若需使用獨立環境，請先執行 install_env.bat)
)

echo.
:: 執行 Python 程式並進入 REPL 模式
python main.py repl

:: 如果程式意外關閉或使用者輸入 exit 離開，暫停視窗讓使用者看清楚最後的訊息
echo.
pause