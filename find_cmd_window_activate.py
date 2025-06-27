import psutil
import ctypes
import sys
import os  # 新增导入
import ctypes.wintypes

def is_admin():
    """检查是否以管理员权限运行（Windows 专用）"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# 新增窗口操作函数
def activate_cmd_window(pid):
    """激活指定PID进程的窗口
    Args:
        pid: 要激活窗口的进程ID
    """
    # 定义Windows API类型
    HWND = ctypes.wintypes.HWND  # 窗口句柄类型
    DWORD = ctypes.wintypes.DWORD  # 32位无符号整数类型

    # 定义窗口枚举回调函数
    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, HWND, ctypes.wintypes.LPARAM)
    def enum_windows_proc(hwnd, lParam):
        """窗口枚举回调函数，用于查找指定PID的窗口"""
        # 从LPARAM参数中提取目标PID（进程ID）
        target_pid = ctypes.cast(lParam, ctypes.POINTER(ctypes.wintypes.DWORD)).contents.value

        # 获取当前窗口所属进程ID
        process_id = DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(
            hwnd,         # 当前窗口句柄
            ctypes.byref(process_id)  # 接收进程ID的输出参数
        )

        # 如果找到目标进程的窗口
        if process_id.value == target_pid:
            # 恢复窗口显示（如果窗口最小化则最大化显示）
            # 9对应Windows API常量SW_RESTORE，用于恢复窗口
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            
            # 将窗口置顶并获取焦点
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            
            return False  # 返回False停止窗口枚举
        return True  # 继续枚举下一个窗口

    # 将PID包装为DWORD类型
    target_pid = DWORD(pid)
    
    # 执行窗口枚举操作，传递回调函数和PID参数
    # EnumWindows会遍历所有顶层窗口，直到回调函数返回False停止
    ctypes.windll.user32.EnumWindows(
        enum_windows_proc,       # 枚举回调函数
        ctypes.byref(target_pid) # 传递目标PID参数
    )

def find_cmd_window_activate():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['pid'] == current_pid:
                continue
                  
            # 激活关联的cmd或Windows Terminal父进程窗口
            parent = proc.parent()
            if parent and parent.name() in ('cmd.exe', 'WindowsTerminal.exe' , 'OpenConsole.exe'):
                try:
                    print(f"Activating window for PID {parent.pid} ({parent.name()})")
                    activate_cmd_window(parent.pid)  # 调用窗口激活函数
                except Exception as e:
                    print(f"窗口激活失败: {str(e)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue



if __name__ == "__main__":
    find_cmd_window_activate()