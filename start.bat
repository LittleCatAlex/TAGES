@echo off
:: 設定編碼為UTF-8
chcp 65001 >nul
title TAGES系統終端機
color 0A

echo =====================================================
echo  正在啟動 TAGES (Taiwan AI Geological Exploration System)...
echo =====================================================
echo.

:: 執行 Python 程式並進入 REPL 模式
python main.py repl

:: 如果程式意外關閉或使用者輸入 exit 離開，暫停視窗讓使用者看清楚最後的訊息
echo.
pause