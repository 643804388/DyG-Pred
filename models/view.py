import matplotlib.pyplot as plt
import numpy as np
import os




plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()  # 添加颜色条
plt.show()
# filepath = './view/STDyn-sgnn3/instanceview{}'.format(s)
# if not os.path.isdir(filepath):
#     # 创建文件夹
#     os.mkdir(filepath)
# plt.savefig('./view/STDyn-sgnn3/instanceview{}/sch{}.jpg'.format(s, s), format='jpg')
plt.close()
