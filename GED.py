import networkx as nx

def graph_edit_distance(graph1, graph2):
    # 创建节点到索引的映射
    node_index_map1 = {node: i for i, node in enumerate(graph1.nodes())}
    node_index_map2 = {node: i for i, node in enumerate(graph2.nodes())}

    # 创建图的邻接矩阵
    adj_matrix1 = nx.adjacency_matrix(graph1).toarray()
    adj_matrix2 = nx.adjacency_matrix(graph2).toarray()

    # 计算节点差异
    node_diff = abs(len(graph1.nodes()) - len(graph2.nodes()))

    # 计算边差异
    edge_diff = abs(adj_matrix1.sum() - adj_matrix2.sum())

    return node_diff + edge_diff

# 示例用法
graph1 = nx.Graph([(1, 2), (2, 3), (3, 4)])
graph2 = nx.Graph([(1, 2), (1, 3), (2, 4)])

ged = graph_edit_distance(graph1, graph2)
print("Graph Edit Distance:", ged)
