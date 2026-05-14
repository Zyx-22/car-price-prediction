@echo off
chcp 65001 >nul
echo ============================================
echo    二手车价格预测系统 - 启动服务
echo ============================================
echo.
echo 正在启动 Streamlit 服务...
echo 访问地址: http://localhost:8501
echo.
echo 按 Ctrl+C 可停止服务
echo ============================================
echo.

:: 激活 Anaconda 环境（如果不在 PATH 中）
call D:\mlp\anaconda3\Scripts\activate.bat

:: 启动 Streamlit
streamlit run "%~dp0app.py" --server.port 8501 --server.headless true

pause
