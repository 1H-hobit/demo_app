```
import pandas as pd

# 读取文件
excel_file = pd.ExcelFile('摄影相机清单.xls')

# 获取所有表名
sheet_names = excel_file.sheet_names

# 遍历不同工作表
for sheet_name in sheet_names:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet_name)

    # 移除包含 'Unnamed' 的列
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=cols_to_drop)

    # 移除包含 nan 的行
    df = df.dropna(how='all')

    # 查看数据的基本信息
    print(f'表名为 {sheet_name} 的基本信息')
    print(df.info())
