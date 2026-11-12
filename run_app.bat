@echo off
cd /d "%~dp0"
title Financial Fraud Network Analytics Dashboard
echo ======================================================================
echo Launching Financial Fraud Network Analytics Dashboard...
echo Local URL: http://localhost:8501
echo ======================================================================
python -m streamlit run app.py --server.port 8501 --server.headless false
pause
