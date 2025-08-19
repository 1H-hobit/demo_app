@echo off
REM 启动 llama-server.exe 在新窗口中运行，并保持窗口打开
echo Starting llama-server in a new window...
start "llama-server" cmd /k "call D:\chainlit\llama\llama-server.exe --device CUDA0 -m D:/chainlit/models/gpt-oss-20b-mxfp4.gguf --jinja --host 127.0.0.1 --port 8181 --ctx-size 10000 --batch-size 5000 --ubatch-size 2048 --mlock --no-perf --gpu-layers 50 --threads 8 --pooling mean --flash_attn --reasoning-format none"

REM 等待几秒钟以确保服务器启动完成
timeout /t 5 >nul

REM 打开浏览器并访问 http://127.0.0.1:8181
start http://127.0.0.1:8181

echo All tasks completed.
exit

