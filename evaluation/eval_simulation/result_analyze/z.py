import numpy as np

# 样本总数
n = 200
# 成功的次数
x = 130

# 计算样本比例
p_hat = x / n

# 95% 置信水平对应的 Z 值
z = 1.96

# 计算标准误差
se = np.sqrt(p_hat * (1 - p_hat) / n)

numb = z * se
print("z = ", numb)

# 计算置信区间的下限和上限
lower_bound = p_hat - numb
upper_bound = p_hat + numb

print(f"样本成功率: {p_hat * 100:.2f}%")
print(f"95% 置信区间: [{lower_bound * 100:.2f}%, {upper_bound * 100:.2f}%]")