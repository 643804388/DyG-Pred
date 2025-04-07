import numpy as np
import matplotlib.pyplot as plt

y_values_decreasing = np.array([
    [0.98, 0.94, 0.86, 0.83, 0.76],
    [0.97, 0.88, 0.79, 0.65, 0.50],
    [0.96, 0.91, 0.75, 0.55, 0.40],
    [0.94, 0.87, 0.80, 0.60, 0.48],
    [0.92, 0.85, 0.78, 0.58, 0.41],
    [0.91, 0.83, 0.72, 0.53, 0.39],
    [0.89, 0.81, 0.69, 0.50, 0.35],
    [0.88, 0.79, 0.66, 0.48, 0.33],
    [0.85, 0.75, 0.63, 0.45, 0.30]
])
# x 轴坐标
x = np.arange(1, 6)  # x轴: 1到5


# 画折线图
plt.figure(figsize=(8, 6))
for i in range(9):
    plt.plot(x, y_values_decreasing[i], marker='o', label=f'Line {i+1}')

# 添加标题和标签
plt.title("Line Plot with 9 Decreasing Trend Lines")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()
plt.grid(True)

# 显示图像
plt.show()


