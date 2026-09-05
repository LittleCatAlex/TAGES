@echo off
chcp 65001 >nul
title TAGES環境安裝程式
color 0B

echo ===================================================
echo   TAGES (Taiwan AI Geological Exploration System)
echo   Python 執行環境與相依套件自動安裝程式
echo ===================================================
echo.

:: 檢查是否安裝了 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 找不到 Python，請確認已安裝 Python 並加入系統環境變數(PATH)。
    echo.
    pause
    exit /b
)

echo [準備] Python 環境檢查通過。
echo [執行] 開始安裝並升級核心套件 pip...
python -m pip install --upgrade pip

echo.
echo [執行] 開始根據 requirements.txt 安裝專案所需套件...
echo.

:: 檢查 requirements.txt 是否存在
if not exist requirements.txt (
    echo [錯誤] 找不到 requirements.txt，請確認檔案與此批次檔放在同一個資料夾。
    echo.
    pause
    exit /b
)

:: 執行安裝
python -m pip install -r requirements.txt

echo.
echo ===================================================
if %errorlevel% equ 0 (
    echo [成功] 所有套件已安裝完畢。
    echo [提示] 現在您可以執行start.bat來啟動系統。
) else (
    echo [警告] 安裝過程中發生錯誤。
)
echo ===================================================
echo.
pause