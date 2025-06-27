import psutil
import ctypes
import sys
import os  # 新增导入
import logging
from typing import List, Set

def is_admin():
    """检查是否以管理员权限运行（Windows 专用）"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def kill_non_comfyui_python(
    allowed_processes: List[str] = None,
    terminate_parent_cmd: bool = True,
    dry_run: bool = False
) -> None:
    """
    终止非ComfyUI的Python和minio进程
    
    参数:
    - allowed_processes: 允许的进程路径关键字列表
    - terminate_parent_cmd: 是否终止父CMD进程
    - dry_run: 只打印不执行
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置默认允许的进程
    if allowed_processes is None:
        allowed_processes = ['comfyui', 'ballontrans_pylibs_win']
    
    current_pid = os.getpid()
    target_processes = {'python.exe', 'pythonw.exe', 'minio.exe'}
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            # 跳过当前进程
            if proc.info['pid'] == current_pid:
                continue
                
            # 检查进程名是否为目标进程
            if proc.info['name'].lower() in target_processes:
                proc_exe = proc.info['exe'].lower() if proc.info['exe'] else ''
                
                # 检查是否在允许列表中
                if not any(keyword in proc_exe for keyword in allowed_processes):
                    log_msg = f"Terminating PID {proc.info['pid']}: {proc_exe}"
                    logging.info(log_msg)
                    
                    if not dry_run:
                        proc.kill()
                    
                    # 终止关联的cmd父进程
                    if terminate_parent_cmd and not dry_run:
                        parent = proc.parent()
                        if parent and parent.name().lower() == 'cmd.exe':
                            log_msg = f"Terminating parent CMD PID {parent.pid}"
                            logging.info(log_msg)
                            parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            logging.debug(f"Skipping process {proc.info['pid']}: {e}")



if __name__ == "__main__":
    if not is_admin():
        print("请以管理员身份运行此脚本！")
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    else:
        kill_non_comfyui_python(
        allowed_processes=['comfyui', 'ballontrans_pylibs_win'],
        terminate_parent_cmd=True,
        dry_run=False  # 设置为True可进行安全测试
    )