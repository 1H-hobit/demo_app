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
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    # 移除包含 nan 的行
    df = df.dropna(how='all')

    # 查看数据的基本信息
    print(f'sheet表名为{sheet_name}的基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 100 and columns < 20:
        # 短表数据（行数少于100且列数少于20）查看全量数据信息
        print(f'sheet表名为{sheet_name}的全部内容信息：')
        print(df.to_csv(sep=    , na_rep='nan'))
    else:

        # 长表数据查看数据前几行信息
        print(f'sheet表名为{sheet_name}的前几行内容信息：')
        print(df.head().to_csv(sep=     , na_rep='nan'))