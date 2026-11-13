@echo off
chcp 65001 >nul
cd /d %~dp0
echo 今の版を起動します。終わったらこの黒い画面を閉じてください。
start "" http://localhost:8000/free_bench.html
py -m http.server 8000
