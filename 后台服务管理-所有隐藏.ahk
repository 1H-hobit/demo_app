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

; 全局变量存储控制台窗口状态
global ConsoleHwnds := []  ; 改为数组存储多个句柄
global IsConsoleVisible := false  ; 初始状态为隐藏
global CurrentConsoleMenuName := "显示控制台"

; 静默启动批处理文件
Run, % ComSpec " /c ""D:\chainlit\chainlit-datalayer\demo_app\运行命令-图片识别-隐式单窗口.bat""", , Hide UseErrorLevel
if ErrorLevel
    TrayTip 错误, 服务启动失败! 错误代码: %ErrorLevel%, 3, 3

; 启动窗口捕获定时器
SetTimer, CaptureConsoleWindow, 500

; 托盘配置（已移除重启服务相关项）
TrayTip, 后台服务, 所有服务已启动, 1, 1
Menu, Tray, NoStandard
Menu, Tray, Add, %CurrentConsoleMenuName%, ToggleConsole
Menu, Tray, Add, 退出服务, ExitApp
Menu, Tray, Tip, 后台服务管理器
Return

; 添加快捷键 Alt+` 切换控制台
!`::
    Goto ToggleConsole
return

; 启动窗口捕获定时器
SetTimer, CaptureConsoleWindow, 500
Return

; 捕获所有DebugConsole窗口
CaptureConsoleWindow:
    ; 保存当前TitleMatchMode并设置为正则表达式模式
    currentMode := A_TitleMatchMode
    SetTitleMatchMode, RegEx
    ; 获取所有ConsoleWindowClass或CASCADIA_HOSTING_WINDOW_CLASS窗口句柄
    WinGet, AllHwnds, List, ahk_class ^(ConsoleWindowClass|CASCADIA_HOSTING_WINDOW_CLASS)$
    ; 恢复原有的TitleMatchMode
    SetTitleMatchMode, %currentMode%
    Loop % AllHwnds {
        hwnd := AllHwnds%A_Index%
        ; 确保窗口未记录且不是脚本自身窗口
        if (hwnd != A_ScriptHwnd && !IsInArray(ConsoleHwnds, hwnd)) {
            ConsoleHwnds.Push(hwnd)
            WinHide, ahk_id %hwnd%  ; 初始隐藏
        }
    }
return

; 切换所有控制台可见性
ToggleConsole:
    IsConsoleVisible := !IsConsoleVisible
    if (IsConsoleVisible) {
        ; 显示控制台时执行特定命令
        RunWait, % ComSpec " /c ""call conda activate D:\chainlit\molmo_env && python ""D:\chainlit\chainlit-datalayer\demo_app\find_cmd_window_activate.py""""", , Hide
    }
    for index, hwnd in ConsoleHwnds {
        if (IsConsoleVisible) {
            WinShow, ahk_id %hwnd%
            WinActivate, ahk_id %hwnd%
        } else {
            WinHide, ahk_id %hwnd%
        }
    }
    ; 更新菜单名称
    NewMenuName := IsConsoleVisible ? "隐藏控制台" : "显示控制台"
    Menu, Tray, Rename, %CurrentConsoleMenuName%, %NewMenuName%
    CurrentConsoleMenuName := NewMenuName
return

; 退出服务
ExitApp:
    RunWait, % ComSpec " /c ""call conda activate D:\chainlit\molmo_env && python ""D:\chainlit\chainlit-datalayer\demo_app\kill_python.py""""", , Hide
    Sleep 1000
    ExitApp
Return

; 辅助函数：检查元素是否在数组中
IsInArray(arr, val) {
    for index, value in arr {
        if (value = val)
            return true
    }
    return false
}