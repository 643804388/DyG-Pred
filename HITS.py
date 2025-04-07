import torch


def hits_algorithm(adjacency_matrix, num_iterations=50):
    num_nodes = adjacency_matrix.shape[0]
    authority_scores = torch.ones(num_nodes)
    hub_scores = torch.ones(num_nodes)

    for _ in range(num_iterations):
        # 更新权威得分
        hub_scores = torch.matmul(adjacency_matrix.t(), authority_scores)
        # 归一化
        hub_scores /= torch.norm(hub_scores, p=2)

        # 更新枢纽得分
        authority_scores = torch.matmul(adjacency_matrix, hub_scores)
        # 归一化
        authority_scores /= torch.norm(authority_scores, p=2)

    return authority_scores, hub_scores


# 示例输入：邻接矩阵
adjacency_matrix = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=torch.float32)

authority_scores, hub_scores = hits_algorithm(adjacency_matrix)
print("Authority Scores:", authority_scores)
print("Hub Scores:", hub_scores)
