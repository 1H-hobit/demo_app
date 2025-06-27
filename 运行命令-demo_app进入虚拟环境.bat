@echo off

REM 切换到 chainlit-datalayer 目录并激活 conda 环境
echo Activating conda environment and starting Docker Compose...
D: 
cd D:\chainlit\chainlit-datalayer\demo_app
start "Chainlit App" cmd /k "call conda activate D:\chainlit\molmo_env
exit