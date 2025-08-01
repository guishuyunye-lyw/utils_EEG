# --------------------------------------------------
# 只修改a相关的列名，然后保存为CSV文件
# --------------------------------------------------

# 创建一个数据副本
df_a = df.copy()

# 修改a相关的列名
df_a = df_a.rename(columns={
    'a_estimate': 'coef_estimate',
    'a_std_error': 'std_error',
    'a_z_value': 't_value',
    'a_p_value': 'p_value'
})

# 显示前几行结果
print("\n修改列名后的数据前几行:")
print(df_a[['spatiotemporal_point', 'coef_estimate', 'std_error', 
           't_value', 'p_value']].head())

# 检查列名是否修改成功
print("\n修改后的列名:")
print(df_a.columns.tolist())

# --------------------------------------------------
# 保存为CSV文件
# --------------------------------------------------
from datetime import datetime

# 获取当前日期并格式化为中文年月日格式
current_date = datetime.now().strftime("%Y年%m月%d日")

# 构建文件名
output_file = f"D:\\LYW\\pre10\\data\\permutation_summary_between\\summary_results_{current_date}_a_updated.csv"

# 保存为CSV文件
df_a.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n已保存修改列名后的数据到文件:\n{output_file}") 