@echo off
REM 启动 llama-server.exe 在新窗口中运行，并保持窗口打开
echo Starting llama-server in a new window...
start "Language model" cmd /k "call conda activate D:\chainlit\molmo_env && D:\chainlit\llama\llama-server.exe --device CUDA0 -m D:/chainlit/models/qwen2.5-7b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8181 --ctx-size 8192 --gpu-layers 8 --threads 8 --pooling mean --flash_attn"

start "Vector model" cmd /k "call conda activate D:\chainlit\molmo_env && D:\chainlit\llama\llama-server.exe --device CUDA0 -m D:/chainlit/models/bge-m3-F16.gguf --host 127.0.0.1 --port 9191 --ctx-size 8192 --gpu-layers 5 --threads 4 --batch-size 8192 --ubatch-size 8192 --pooling mean --flash_attn --no-warmup"


echo All tasks completed.
exit
