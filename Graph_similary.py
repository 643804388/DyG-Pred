import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

# 创建示例图
graph1 = nx.Graph()
graph1.add_nodes_from([1, 2, 3])
graph1.add_edges_from([(1, 2), (2, 3)])

graph2 = nx.Graph()
graph2.add_nodes_from([1, 2, 3])
graph2.add_edges_from([(1, 2), (1, 3)])

# 提取图结构特征
def extract_graph_structure_feature(graph):
    adjacency_matrix = nx.to_numpy_matrix(graph)
    return adjacency_matrix.flatten()

graph1_structure_feature = extract_graph_structure_feature(graph1)
graph2_structure_feature = extract_graph_structure_feature(graph2)

# 提取节点属性特征
# 这里假设节点属性为节点的度数
def extract_node_attribute_feature(graph):
    degrees = np.array(list(dict(graph.degree()).values()))
    return degrees

graph1_attribute_feature = extract_node_attribute_feature(graph1)
graph2_attribute_feature = extract_node_attribute_feature(graph2)

# 计算相似性得分
structure_similarity_score = cosine(graph1_structure_feature, graph2_structure_feature)
attribute_similarity_score = cosine(graph1_attribute_feature, graph2_attribute_feature)
similarity_score = 1 - (structure_similarity_score + attribute_similarity_score) / 2

print("Graph Similarity Score:", similarity_score)
