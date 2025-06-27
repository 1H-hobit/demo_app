@echo off
REM 切换到 chainlit-datalayer 目录并激活 conda 环境
echo Activating conda environment
D:
cd D:\chainlit\chainlit-datalayer
call conda activate D:\chainlit\molmo_env

REM 启动minio服务（后台运行）使用指定窗口标题
echo Starting minio app...
cd D:\chainlit\chainlit-datalayer\demo_app
start "DebugConsole" /B cmd /c "call conda activate D:\chainlit\molmo_env && minio server D:\chainlit\chainlit-datalayer\demo_app --console-address :9001"

REM 启动lightrag_server（后台运行）使用相同窗口标题
echo Starting lightrag_server...
start "DebugConsole" /B cmd /c "call conda activate D:\chainlit\molmo_env && python D:\chainlit\chainlit-datalayer\demo_app\lightrag_server.py"

REM 启动Chainlit应用（后台运行）使用相同窗口标题
echo Starting Chainlit app...
start "DebugConsole" cmd /c "call conda activate D:\chainlit\molmo_env && python -m chainlit run D:\chainlit\chainlit-datalayer\demo_app\app.py"

echo All tasks completed.
exit