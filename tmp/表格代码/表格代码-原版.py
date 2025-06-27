```py
import pandas as pd

# 读取文件
excel_file = pd.ExcelFile('摄影相机清单.xls')

# 获取所有表名
sheet_names = excel_file.sheet_names

# 遍历不同工作表
for sheet_name in sheet_names:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet_name)
    
    # 将列名转换为字符串类型（关键修复）
    df.columns = df.columns.astype(str)
    
    # 移除包含'Unnamed'的列
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    
    # 移除包含nan的行（保留至少有一个有效值的行）
    df = df.dropna(how='all')
    
    # 查看数据的基本信息
    print(f'sheet表名为{sheet_name}的基本信息：')
    df.info()
    
    # 查看数据集行数和列数
    rows, columns = df.shape
    
    if rows < 100 and columns < 20:
        print(f'sheet表名为{sheet_name}的全部内容信息：')
        # 使用制表符分隔，空值显示为nan
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        print(f'sheet表名为{sheet_name}的前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))



```py
import pandas as pd

# 读取文件
excel_file = pd.ExcelFile('摄影相机清单.xls')

# 获取所有表名
sheet_names = excel_file.sheet_names

# 初始化一个空字典来存储每个工作表的统计数据
stats = {}

# 遍历不同工作表
for sheet_name in sheet_names:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet_name)

    # 查看数据的基本信息
    print(f'sheet表名为{sheet_name}的基本信息：')
    print(df.info())

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 100 and columns < 20:
        print(f'sheet表名为{sheet_name}的全部内容信息：')
        # 使用制表符分隔，空值显示为nan
        print(df.to_csv(sep='\t', na_rep='nan'))

    else:
        print(f'sheet表名为{sheet_name}的前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))

    # 查看缺失值情况
    print(f'sheet表名为{sheet_name} 的缺失值情况：')
    print(df.isnull().sum())

    # 计算总价格、平均价格、最小价格和最大价格
    print(f'Sheet1 的统计数据：')
    if sheet_name == 'Sheet1':
        print(f"总价格: {df['价格'].sum()}")
        print(f"平均价格: {df['价格'].mean()}")
        print(f"最小价格: {df['价格'].min()}")
        print(f"最大价格: {df['价格'].max()}")

    # 将统计数据存储在字典中
    stats[sheet_name] = {
        '总价格': df['价格'].sum() if sheet_name == 'Sheet1' else None,
        '平均价格': df['价格'].mean() if sheet_name == 'Sheet1' else None,
        '最小价格': df['价格'].min() if sheet_name == 'Sheet1' else None,
        '最大价格': df['价格'].max() if sheet_name == 'Sheet1' else None
    }



```py
import pandas as pd

# 读取文件
excel_file = pd.ExcelFile('摄影相机清单.xls')

# 获取所有表名
sheet_names = excel_file.sheet_names

# 初始化总价格
total_price = 0

# 遍历不同工作表
for sheet_name in sheet_names:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet_name)

    # 【修复点】将列名转为字符串后再过滤
    # 移除包含'Unnamed'的列
    df = df.loc[:, ~df.columns.astype(str).str.contains('Unnamed')]

    # 移除包含nan的行（保留至少有一个有效值的行）
    df = df.dropna(how='all')

    # 查看数据的基本信息
    df.info()

    # 计算总价格
    if '价格' in df.columns:
        total_price += df['价格'].sum()

        # 查看数据集行数和列数
        rows, columns = df.shape

        if rows < 100 and columns < 20:
            print(f'sheet表名为{sheet_name}的全部内容信息：')
            # 使用制表符分隔，空值显示为nan
            print(df.to_csv(sep='\t', na_rep='nan'))
        else:
            print(f'sheet表名为{sheet_name}的前几行内容信息：')
            print(df.head().to_csv(sep='\t', na_rep='nan'))

# 【修复点】总价格输出放在循环外，避免重复打印
print(f'总价格为: {total_price}')


```py
import pandas as pd

def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        return df.head().to_string()  # 仅输出前几行
    except FileNotFoundError:
        return "错误：文件未找到。"
    except Exception as e:
        return f"错误：{str(e)}"

file_path = '摄影相机清单.xls'
content = read_excel_file(file_path)
print(f'内容为:\n{content}')