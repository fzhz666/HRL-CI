import numpy as np
import matplotlib.pyplot as plt

# 组名
group_names = ['DDPG', 'pH-DRL', 'HRL-CI']

# 数据
success_rates = np.array([58.5, 84, 86])
collision_rates = np.array([6, 8, 3.5])
failure_rates = np.array([35.5, 8, 10.5])

# # 组名 task2
# group_names = ['DDPG', 'DDPG*', 'pH-DRL', 'HRL-CI']
#
# # 数据
# success_rates = np.array([36.5, 41, 55.5, 79])
# collision_rates = np.array([7, 4, 11.5, 3.5])
# failure_rates = np.array([56.5, 55, 33, 17.5])

# 颜色调整为低对比度风格
colors = ['#76c7c0', '#f2a1a1', '#f3e5ab']  # '#d3a4ff'、#ffb347

# 绘制柱状图
fig, ax = plt.subplots(figsize=(8, 6), edgecolor='none')
bar_width = 0.3
x = np.arange(len(group_names))

# 底部起始位置
bottoms = np.zeros(len(group_names))

# 绘制每个部分
for rates, color, label in zip([success_rates, collision_rates, failure_rates], colors, ['success', 'collision', 'overtime']):
    bars = ax.bar(x, rates, bar_width, bottom=bottoms, color=color, label=label, edgecolor='none')
    bottoms += rates  # 叠加高度
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{height:.1f}', ha='center', va='center', fontsize=22, color='black')

# 设置横纵坐标
ax.set_xticks(x)
ax.set_xticklabels(group_names, fontsize=28, fontweight='bold')  # 横坐标字体变大
ax.set_yticks(np.arange(0, 101, 10))
ax.set_ylabel('percentage (%)', fontsize=28, fontweight='bold')

# 只保留下方的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)

# 添加图例并放置在上方
ax.legend(fontsize=28, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, edgecolor='none')

# 显示图形
plt.show()
