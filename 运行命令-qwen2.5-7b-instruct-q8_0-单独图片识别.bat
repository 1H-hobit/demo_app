@echo off
REM 启动 llama-server.exe 在新窗口中运行，并保持窗口打开
echo Starting llama-server in a new window...
start "Language model" cmd /k "call D:\chainlit\llama\llama-server.exe --device CUDA0 -m D:/chainlit/models/qwen2.5-7b-instruct-q8_0.gguf --host 127.0.0.1 --port 8181 --ctx-size 8192 --batch-size 2048 --ubatch-size 1024 --mlock --no-perf --gpu-layers 50 --threads 8 --pooling mean --flash_attn"

REM 等待几秒钟以确保服务器启动完成
timeout /t 5 >nul

REM 打开浏览器并访问 http://127.0.0.1:8181
start http://127.0.0.1:8181

echo All tasks completed.
exit
