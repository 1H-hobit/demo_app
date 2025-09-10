import os
import re

def natural_sort_key(s):
    """
    生成自然排序键（数字部分按数值大小排序）
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def sort_filenames(files, sort_method="natural"):
    """
    根据指定方法对文件名进行排序
    
    参数:
    files -- 文件名列表
    sort_method -- 排序方法: "natural", "alphabetical", "reverse", "by_size", "by_modified_time"
    """
    if sort_method == "natural":
        return sorted(files, key=natural_sort_key)
    elif sort_method == "alphabetical":
        return sorted(files)
    elif sort_method == "reverse":
        return sorted(files, reverse=True)
    elif sort_method == "by_size":
        return sorted(files, key=lambda f: os.path.getsize(f))
    elif sort_method == "by_modified_time":
        return sorted(files, key=lambda f: os.path.getmtime(f))
    else:
        return files

def is_excluded_file(filename):
    """
    检查文件是否应该被排除（.py和.bat文件）
    """
    excluded_extensions = ['.py', '.bat']
    _, ext = os.path.splitext(filename)
    return ext.lower() in excluded_extensions

def main():
    print("=== 文件名自定义命名排序工具 ===\n")
    print(f"当前目录: {os.getcwd()}")
    print("注意：.py和.bat文件将不会被重命名\n")
    
    # 获取当前目录
    directory = "."
    
    # 获取所有文件（排除.py和.bat）
    try:
        all_files = [f for f in os.listdir(directory) 
                    if os.path.isfile(os.path.join(directory, f)) and not is_excluded_file(f)]
    except PermissionError:
        print("错误：没有权限访问该目录！")
        return
    
    if not all_files:
        print("当前目录中没有可处理的文件（已排除.py和.bat文件）。")
        return
    
    print(f"\n找到 {len(all_files)} 个可处理文件（已排除.py和.bat文件）。")
    
    # 选择排序方法
    print("\n请选择排序方法:")
    print("1. 自然排序 (默认)")
    print("2. 字母顺序")
    print("3. 反向排序")
    print("4. 按文件大小")
    print("5. 按修改时间")
    
    choice = input("请输入选项 (1-5): ").strip()
    
    sort_methods = {
        "1": "natural",
        "2": "alphabetical",
        "3": "reverse",
        "4": "by_size",
        "5": "by_modified_time"
    }
    
    sort_method = sort_methods.get(choice, "natural")
    
    # 排序文件
    sorted_files = sort_filenames(all_files, sort_method)
    
    # 显示排序结果
    print(f"\n排序结果 ({sort_method}):")
    print("-" * 50)
    for i, filename in enumerate(sorted_files, 1):
        print(f"{i:3d}. {filename}")
    
    # 询问是否重命名
    rename = input("\n是否要按照此顺序重命名文件？(y/N): ").strip().lower()
    
    if rename == 'y':
        # 获取文件名前缀
        prefix = input("请输入文件名前缀（默认为空）: ").strip()
        
        # 默认保留原扩展名，不再询问
        keep_ext = True
        
        # 确认操作
        confirm = input("确认执行重命名操作？此操作不可撤销！(y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消。")
            return
        
        # 执行重命名
        success_count = 0
        for i, filename in enumerate(sorted_files, 1):
            # 再次检查是否为排除文件（以防万一）
            if is_excluded_file(filename):
                print(f"跳过排除文件: {filename}")
                continue
                
            # 获取文件扩展名（默认保留原扩展名）
            name, ext = os.path.splitext(filename)
            if not ext:  # 如果没有扩展名
                ext = ""
            
            # 构建新文件名
            new_name = f"{prefix}{i:03d}{ext}"
            
            # 重命名文件
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                # 检查新文件名是否已存在
                if os.path.exists(new_path):
                    print(f"警告: 目标文件已存在，跳过 {filename}")
                    continue
                    
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_name}")
                success_count += 1
            except Exception as e:
                print(f"错误: 无法重命名 {filename} -> {new_name}: {str(e)}")
        
        print(f"\n操作完成！成功重命名 {success_count} 个文件。")

if __name__ == "__main__":
    main()