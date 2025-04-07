import numpy as np
import chardet
import csv

# Open file
fileHandler  =  open  ("F:\动态图生成\ST-DyEnt-S3-2\data\DataSet-IJCNLP2011\DataSet-IJCNLP2011\data sets/New York Times.txt",  "r")
# Get list of all lines in file
listOfLines  =  fileHandler.readlines()
# Close file
fileHandler.close()
current_event = []
events = []
for line in listOfLines:
    line = line.strip()
    if line.startswith('NYT-'):
        if current_event:
            events.append(current_event)
            current_event = []
    elif line:
        current_event.append(line.split('\t'))

if current_event:
    events.append(current_event)

for event in events:
    print(event)

# 加载.npy文件
# file_path = r'F:\动态图生成\SpikeNet-master\SpikeNet-master\data\dblp\dblp.npy'  # 替换为实际的.npy文件路径
# data = np.load(file_path)
# print()
# import numpy as np
#
#
# file_path = r'F:\动态图生成\data_ConstructingNEEG_IJCAI_2018\data\corpus_index_train0_with_args_all_chain.data'
# try:
#     data_1 = np.load(file_path, allow_pickle=True)
#     print(data)
# except FileNotFoundError:
#     print(f"错误：未找到文件 {file_path}。")
# except Exception as e:
#     print(f"发生未知错误: {e}")
# 打开.data文件以读取模式

# 检测文件编码
# with open(r'F:\动态图生成\dataset_large\vocab_index_test.data', 'rb') as f:
#     raw_data = f.read()
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']

# 使用检测到的编码打开文件
# with open(r'F:\动态图生成\dataset_large\vocab_index_test.data', 'rb') as f:
#     content = f.read()
