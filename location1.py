import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义参数
R = 1  # 半径
beta = 70
theta = 40

beta = beta * np.pi / 180
theta = theta * np.pi / 180

d = R * np.cos(beta)  # 计算d的值

# 创建φ值
phi = np.linspace(0, 2 * np.pi, 100)

# 计算x, y, z
x = R * np.sin(beta) * np.sin(phi)
y = R * np.sin(beta) * np.cos(phi) * np.sin(theta) + d * np.cos(theta)
z = R * np.sin(beta) * np.cos(phi) * np.cos(theta) - d * np.sin(theta)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原点
ax.scatter([0], [0], [0], color="g", label="Origin")

# 绘制基础平面
circle_x = R * np.cos(phi)
circle_y = R * np.sin(phi)
ax.plot(circle_x, circle_y, 0, color='lightblue')

# 绘制轨迹，颜色根据z的值确定
for i in range(len(z)):
    if z[i] > 0:
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color='red')
    else:
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color='blue')

# 显示图例
red_patch = plt.plot([], [],
                     marker="o",
                     ms=10,
                     ls="",
                     mec=None,
                     color='red',
                     label="Day")[0]
blue_patch = plt.plot([], [],
                      marker="o",
                      ms=10,
                      ls="",
                      mec=None,
                      color='blue',
                      label="Night")[0]
plt.legend(handles=[red_patch, blue_patch], loc='upper right')

# 显示图形
plt.show()