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
    
    # 按文件扩展名分组
    file_groups = {}
    for filename in all_files:
        _, ext = os.path.splitext(filename)
        if ext not in file_groups:
            file_groups[ext] = []
        file_groups[ext].append(filename)
    
    # 对每个组内的文件进行排序
    sorted_groups = {}
    for ext, files in file_groups.items():
        sorted_groups[ext] = sort_filenames(files, sort_method)
    
    # 显示排序结果
    print(f"\n排序结果 ({sort_method}):")
    print("-" * 50)
    for ext, files in sorted_groups.items():
        print(f"\n扩展名 {ext if ext else '[无扩展名]'}:")
        for i, filename in enumerate(files, 1):
            print(f"  {i:3d}. {filename}")
    
    # 询问是否重命名（默认改为y）
    rename = input("\n是否要按照此顺序重命名文件？(Y/n，默认为y): ").strip().lower()
    if rename == '':
        rename = 'y'
    
    if rename == 'y':
        # 获取文件名前缀
        prefix = input("请输入文件名前缀（默认为空）: ").strip()
        
        # 获取文件名后缀
        suffix = input("请输入文件名后缀（默认为空）: ").strip()
        
        # 生成重命名预览
        print("\n预览重命名结果:")
        print("-" * 50)
        rename_map = {}  # 存储旧文件名到新文件名的映射
        
        for ext, files in sorted_groups.items():
            print(f"\n扩展名 {ext if ext else '[无扩展名]'}:")
            for i, filename in enumerate(files, 1):
                # 再次检查是否为排除文件（以防万一）
                if is_excluded_file(filename):
                    print(f"  跳过排除文件: {filename}")
                    continue
                    
                # 构建新文件名（添加后缀支持）
                new_name = f"{prefix}{i:03d}{suffix}{ext}"
                rename_map[filename] = new_name
                print(f"  {filename} -> {new_name}")
        
        # 检查是否有文件名冲突
        conflicts = []
        new_names = list(rename_map.values())
        for old_name, new_name in rename_map.items():
            if new_names.count(new_name) > 1:
                conflicts.append(new_name)
        
        if conflicts:
            print(f"\n警告: 检测到 {len(set(conflicts))} 个文件名冲突!")
            for conflict in set(conflicts):
                print(f"  冲突文件名: {conflict}")
            print("重命名操作可能会导致文件覆盖，建议修改前缀或后缀。")
        
        # 确认操作（默认改为y）
        confirm = input("\n确认执行重命名操作？此操作不可撤销！(Y/n，默认为y): ").strip().lower()
        if confirm == '':
            confirm = 'y'
            
        if confirm != 'y':
            print("操作已取消。")
            return
        
        # 执行重命名
        success_count = 0
        for old_name, new_name in rename_map.items():
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            
            try:
                # 检查新文件名是否已存在（除了自己重命名的情况）
                if os.path.exists(new_path) and old_name != new_name:
                    print(f"警告: 目标文件已存在，跳过 {old_name}")
                    continue
                    
                os.rename(old_path, new_path)
                print(f"重命名: {old_name} -> {new_name}")
                success_count += 1
            except Exception as e:
                print(f"错误: 无法重命名 {old_name} -> {new_name}: {str(e)}")
        
        print(f"\n操作完成！成功重命名 {success_count} 个文件。")

if __name__ == "__main__":
    main()