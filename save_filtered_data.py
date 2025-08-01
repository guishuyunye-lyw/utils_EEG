# --------------------------------------------------
# 保存筛选和重命名后的数据到CSV文件
# --------------------------------------------------
from datetime import datetime

# 获取当前日期并格式化为中文年月日格式
current_date = datetime.now().strftime("%Y年%m月%d日")

# 构建文件名
output_file = f"D:\\LYW\\pre10\\data\\permutation_summary_between\\summary_results_{current_date}_ab_updated.csv"

# 保存为CSV文件
df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n已保存筛选后的数据到文件:\n{output_file}") 