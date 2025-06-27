; 阻止窗口自动最小化
OnMessage(0x0112, Func("PreventAutoMinimize")) ; WM_SYSCOMMAND
OnMessage(0x0005, Func("PreventAutoMinimize")) ; WM_SIZE
OnMessage(0x0018, Func("PreventAutoMinimize"))  ; WM_SHOWWINDOW

PreventAutoMinimize(wParam, lParam, uMsg, hwnd) {
    if (uMsg = 0x0112 && wParam = 0xF020 && hwnd = A_ScriptHwnd) {
        WinMinimize
        return 0
    }
    if (uMsg = 0x0005 && wParam = 1 && hwnd = A_ScriptHwnd)
        return 0
    if (uMsg = 0x0018 && lParam = 1)
        return 0
}

; 静默启动批处理文件（彻底隐藏窗口）
Run, % ComSpec " /c ""D:\chainlit\chainlit-datalayer\demo_app\运行命令-图片识别.bat""", , Hide UseErrorLevel
if ErrorLevel
    TrayTip 错误, 服务启动失败! 错误代码: %ErrorLevel%, 3, 3

; 托盘配置
TrayTip, 后台服务, 所有服务已启动, 1, 1
Menu, Tray, NoStandard
Menu, Tray, Add, 重启服务, RestartApps
Menu, Tray, Add, 退出服务, ExitApp
Menu, Tray, Default, 重启服务
Menu, Tray, Tip, 后台服务管理器
Return

; 重启服务
RestartApps:
    ; 终止进程
    RunWait, % ComSpec " /c taskkill /f /im minio.exe /t", , Hide
    RunWait, % ComSpec " /c taskkill /f /im python.exe /t", , Hide
    RunWait, % ComSpec " /c taskkill /f /im pythonw.exe /t", , Hide
    Sleep 5000
    
    ; 重新启动
    Run, % ComSpec " /c ""D:\chainlit\chainlit-datalayer\demo_app\运行命令-图片识别.bat""", , Hide UseErrorLevel
    if ErrorLevel
        TrayTip, 错误, 重启失败! 错误代码: %ErrorLevel%, 3, 3
    else
        TrayTip, 后台服务, 服务已重启, 1, 1
Return

; 退出服务
ExitApp:
    RunWait, % ComSpec " /c taskkill /f /im minio.exe /t", , Hide
    RunWait, % ComSpec " /c taskkill /f /im python.exe /t", , Hide
    RunWait, % ComSpec " /c taskkill /f /im pythonw.exe /t", , Hide
    Sleep 1000
    ExitApp
Return