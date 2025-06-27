@echo off
REM 切换到 chainlit-datalayer 目录并激活 conda 环境
echo Activating conda environment
D: 
cd D:\chainlit\chainlit-datalayer
call conda activate D:\chainlit\molmo_env

REM 切换到 demo_app 目录并启动 minio 应用
echo Starting minio app...
cd D:\chainlit\chainlit-datalayer\demo_app
start "minio App" cmd /k "call conda activate D:\chainlit\molmo_env && minio server D:\chainlit\chainlit-datalayer\demo_app --console-address ":9001""

REM 自动打开浏览器访问MinIO控制台
echo Opening MinIO console in default browser...
#start http://localhost:9001

REM 切换到 demo_app 目录并启动 lightrag_server
echo Starting lightrag_server...
cd D:\chainlit\chainlit-datalayer\demo_app
start "lightrag_server" cmd /k "call conda activate D:\chainlit\molmo_env && python D:\chainlit\chainlit-datalayer\demo_app\lightrag_server.py"

REM 切换到 demo_app 目录并启动 Chainlit 应用
echo Starting Chainlit app...
cd D:\chainlit\chainlit-datalayer\demo_app
start "Chainlit App" cmd /k "call conda activate D:\chainlit\molmo_env && python -m chainlit run D:\chainlit\chainlit-datalayer\demo_app\app.py"

echo All tasks completed.
exit