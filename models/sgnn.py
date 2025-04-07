import math
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.embedding import myBert
import matplotlib.pyplot as plt
import numpy as np
import os

class SGNN(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, sch_hidden_size, ske_hidden_size):
        super(SGNN, self).__init__()
        self.sch_hidden_size = sch_hidden_size
        self.ske_hidden_size = ske_hidden_size
        self.Bert = myBert()
        self.encoder = Encoder()
        self.Lstm = nn.LSTM(input_size=1024,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            )
        # self.node_choose = nn.Sequential(nn.Linear(128, 64), nn.Dropout(), nn.ReLU(),nn.Linear(64, 8), nn.Dropout(), nn.ReLU(), nn.Softmax())
        self.DIC_conv = GCNLayer(1024, 128)
        self.T_conv = GCNLayer(128, 128)
        self.S_conv = GCNLayer(128, 128)
        self.e = nn.Linear(128, 128)
        self.reduce = nn.Linear(1024, 128)
        self.x = nn.Linear(1024, 128)
        self.reduce = nn.Linear(256, 128)
        # self.schemas_scores = BaseSGNN(8, 128)
        self.q_S = nn.Linear(128, 128)
        self.k_S = nn.Linear(128, 128)
        self.v_S = nn.Linear(128, 128)
        self.layer_norm_S = nn.LayerNorm([128])

        self.q_T = nn.Linear(128, 128)
        self.k_T = nn.Linear(128, 128)
        self.v_T = nn.Linear(128, 128)
        self.layer_norm_T = nn.LayerNorm([128])
        self.Gate_S = nn.Linear(128, 128)
        self.Gate_T = nn.Linear(128, 128)
        self.Fusion = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                      nn.ReLU(),
                                      nn.Dropout(), )
        self.choose = nn.Sequential(nn.Linear(in_features=128, out_features=30522),

                                    nn.Softmax(),
                                    )
        self.eps = 1e-12


        self.relu = nn.Sequential(nn.ReLU())
        self.schemas_S_T = SchemasSGNN(128, sch_hidden_size)


    def set_l(self, l):
        self.schemas_sgnn.set_l(l)
        self.skeleton_sgnn.set_l(l)

    def forward(self, s, batch_size, nodes_sch, A_sch, nodes_ske, A_ske, S_adj, T_adj):
        # 获取图节点表示与静态图邻接矩阵
        batch_list = []
        schadj_list = []
        idx_list = []
        embedding_batch_list = []
        for batch in nodes_sch:
            idx_batch_list = []
            if len(embedding_batch_list) <= 9:
                node_embedding_list = []
                for i in batch:
                    node_embeddings, idx = self.Bert(i)
                    node_embedding_list.append(node_embeddings)
                    idx_batch_list.append(idx)
                idx_target = torch.Tensor(idx_batch_list).to(torch.int64).to(self.device)
                idx_list.append(idx_target)
                embedding_batch = torch.stack(node_embedding_list, dim=0)
                embedding_batch_list.append(embedding_batch)
            else:
                for i in batch:
                    node_embeddings, idx = self.Bert(i)
                    idx_batch_list.append(idx)
                idx_target = torch.Tensor(idx_batch_list).to(torch.int64).to(self.device)
                idx_list.append(idx_target)
                    # idx_target_list.append(idx_target)

        node_embedding = torch.stack(embedding_batch_list, dim=1).squeeze(2)
        # 皮尔逊相关系数作静态图
        corr_sch = []
        for i in range(0, node_embedding.shape[0]):
            corr_sch = (torch.corrcoef(node_embedding[i, :, :]))
            schadj_list.append(corr_sch)
        batch_list.append(node_embedding)
        nodes_sch_embedding = torch.stack(batch_list).squeeze(0)
        sch_adj = torch.stack(schadj_list, dim=0)
        # batch_list = []
        # skeadj_list = []
        # for batch in nodes_ske:
        #     node_embedding_list = []
        #     for node in batch:
        #         node_embeddings = self.Bert(node)
        #         node_embedding_list.append(node_embeddings)
        #     node_embedding = torch.stack(node_embedding_list)
        #     batch_list.append(node_embedding)
        #     corr_sch = torch.corrcoef(node_embedding.squeeze(1))
        #     skeadj_list.append(corr_sch)
        # nodes_ske_embedding = torch.stack(batch_list).squeeze(2)
        # ske_adj = torch.stack(skeadj_list)
        # 作目标矩阵

        targets = idx_list
        nodes_target = torch.tensor([[0],[1],[2],[3],[4],[5],[6],[7]]).to(torch.int64).to(self.device)
        # nodes_sch, A_sch, nodes_ske, A_ske = nodes_sch.to(self.device), A_sch.to(self.device), nodes_ske.to(self.device), A_ske.to(self.device)
        # 数据放入cuda
        nodes_sch_embedding, A_sch, sch_adj = nodes_sch_embedding.to(self.device), A_sch.to(self.device), sch_adj.to(self.device)
        S_adj, T_adj = S_adj.to(self.device), T_adj.to(self.device)
        similarity_matrix_list = []
        nodes_sch_S_T_list = []
        node_id_all = []
        node_choose_loss = 0.
        nodes_sch_S_T = nodes_sch_embedding
        similarity_matrix_sub = torch.zeros_like(sch_adj)
        S_adj_loss = torch.zeros_like(sch_adj)
        accuracy_list = []
        # 事件预测+动态图重构，输入每一时间步加入的节点，输出重构之后的子图
        for i in range(1, nodes_sch_embedding.shape[1]):


            # 准备空矩阵，为子图节点与边表示存储做准备
            node_target = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1]).to(self.device)
            # T_sch_adj = torch.zeros_like(sch_adj).to(self.device)
            # T_adj_sub = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1], nodes_sch_embedding.shape[1]).to(self.device)
            # S_adj_sub = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1], nodes_sch_embedding.shape[1]).to(self.device)
            node_target[:, i] = 1
            block = 1

            node_id, similarity_matrix_sub, nodes_sch_S_T, entry_loss, accuracy_gen = self.encoder(i, similarity_matrix_sub, A_sch, nodes_sch_embedding, sch_adj, T_adj, S_adj, nodes_sch_S_T, idx_list)
            node_id_all.append(node_id)
            accuracy_list.append(accuracy_gen)
            similarity_matrix_list.append(similarity_matrix_sub.clone())
            nodes_sch_S_T_list.append(nodes_sch_S_T)
            node_choose_loss = node_choose_loss + entry_loss

        # 可视化
        # 增量矩阵可视化
        # for t in range(0, len(similarity_matrix_list)):
        #     corr_similarity = torch.corrcoef(nodes_sch_S_T_list[t].squeeze(0))
        #     similarity_matrix_now = similarity_matrix_list[t][:, :t+2, :t+2].squeeze(0)
        #     normalized_tensor = min_max_normalization(similarity_matrix_now)
        #
        #     similarity_all = (corr_similarity + normalized_tensor) / 2
        #
        #     # 将二维张量转换为NumPy数组
        #     array = np.array(similarity_all.to('cpu').detach())
        #
        #     # 使用Matplotlib绘制图形
        #     plt.imshow(array, cmap='viridis', interpolation='nearest')
        #     plt.colorbar()  # 添加颜色条
        #     # plt.show()
        #     filepath = './view/STDyn/instance{}'.format(s)
        #     if not os.path.isdir(filepath):
        #         # 创建文件夹
        #         os.mkdir(filepath)
        #     plt.savefig('./view/STDyn/instance{}/plot{}.jpg'.format(s, t), format='jpg')
        #     plt.close()


        S_adj_ex = S_adj.repeat(similarity_matrix_sub.shape[0], 1, 1)
        graph_loss = F.mse_loss(S_adj_ex, similarity_matrix_sub)

        accuracy_list = torch.stack(accuracy_list)

        # 邻接图评分
        # 邻接矩阵归一化
        # 最小-最大归一化
        similarity_matrix_min_max_normalization = min_max_normalization(similarity_matrix_sub)
        similarity_matrix_min_max_std = torch.std(similarity_matrix_sub)
        similarity_matrix_min_max_mean = torch.mean(similarity_matrix_sub)
        # 对张量进行 Z-Score 标准化
        # normalized_tensor = (similarity_matrix_sub - similarity_matrix_min_max_mean) / similarity_matrix_min_max_std

        graph_matrix = torch.where(similarity_matrix_min_max_normalization > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
        ones_count_list = []
        for i in range(0, graph_matrix.shape[-1]):
            ones_count = (graph_matrix[:, i] == 1).sum().item()
            ones_count_list.append(torch.tensor(ones_count))
        count = torch.stack(ones_count_list).to(self.device).float()
        b = similarity_matrix_sub.shape[0]
        S_count = torch.tensor([1 * b, 2 * b, 2 * b, 2 * b, 2 * b, 2 * b, 2 * b, 2 * b, 2 * b, 1 * b], dtype=torch.float32).to(self.device)
        similarity_matrix_flattened = similarity_matrix_min_max_normalization.view(-1)
        S_adj_flattened = S_adj_ex.view(-1)
        structure_similarity_score = torch.cosine_similarity(similarity_matrix_flattened, S_adj_flattened, dim=0)
        attribute_similarity_score = torch.cosine_similarity(count, S_count, dim=0)
        similarity_score = 1 - (structure_similarity_score + attribute_similarity_score) / 2
        # 图编辑距离
        edge_diff = abs(count.sum() - S_count.sum())


        # node_id_tensor = torch.stack(node_id_all, dim=1)
        # node_target = T_adj[0:-1, :].unsqueeze(0).to(torch.int64)
        # entry_probs = torch.gather(node_id_tensor, dim=-1, index=nodes_target.unsqueeze(0)).squeeze()
        # entry_loss = -torch.log(entry_probs + self.eps)
        # similarity_matrix = torch.stack(similarity_matrix_list, dim=1)
        # similarity_matrix_S_T = torch.mean(similarity_matrix, dim=1)
        # entry_probs = torch.gather(similarity_matrix_sub, dim=-1, index=nodes_target.unsqueeze(0)).squeeze()
        # entry_loss = -torch.log(entry_probs + self.eps)
        # 图嵌入
        # nodes_S_T = torch.mean(torch.stack(nodes_sch_S_T_list, dim=1), dim=1)
        nodes_S_T = nodes_sch_S_T_list[-1]
        nodes_Static = self.x(nodes_sch_embedding)
        # 获取融合图
        S_T_Graph_nodes = torch.concat([nodes_S_T, nodes_Static], dim=-1)
        S_T_Graph_nodes = self.reduce(S_T_Graph_nodes)
        # 融合图与动态图作交叉注意力
        q_s = self.q_S(nodes_S_T)
        k_s = self.k_S(nodes_S_T)
        v_s = self.v_S(S_T_Graph_nodes)
        attn1 = (q_s @ k_s.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        hidden_S = self.layer_norm_S(attn1 @ v_s + nodes_S_T)
        # 融合图与静态图作交叉注意力
        q_t = self.q_T(nodes_Static)
        k_t = self.k_T(nodes_Static)
        v_t = self.v_T(S_T_Graph_nodes)
        attn2 = (q_t @ k_t.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        hidden_T = self.layer_norm_T(attn2 @ v_t + nodes_Static)
        z =  torch.sigmoid(torch.add(hidden_S, hidden_T))
        H = torch.add(torch.mul(z, hidden_S), torch.mul(1 - z, hidden_T))
        P_dist = self.choose(torch.mean(H, dim=1))
        # Loss计算
        probs = torch.gather(P_dist, dim=-1, index=idx_list[-1].unsqueeze(1)).squeeze()
        step_loss = -torch.log(probs + self.eps)
        # entry_loss_pred = entry_loss.mean()
        step_loss_all = step_loss.mean()

        # accuracy计算
        choose = torch.max(P_dist, dim=1)
        choose_idx = choose.indices
        Target = idx_list[-1]
        accuracy_gen = torch.sum(torch.eq(Target, choose_idx)) / nodes_sch_embedding.shape[0]


        return step_loss_all, graph_loss, node_choose_loss, accuracy_gen, structure_similarity_score, attribute_similarity_score, similarity_score, edge_diff, accuracy_list

    def loss_node(self, output, target):
        return self.schemas_sgnn.loss_node(output, target)

    def accu_node(self, output, target):
        return self.schemas_sgnn.accu_node(output, target)

    def loss_path(self, output, target):
        return self.schemas_sgnn.loss_path(output, target)

    def accu_path(self, output, target):
        return self.schemas_sgnn.accu_path(output, target)

class Encoder(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, ):
        super(Encoder, self).__init__()
        self.Lstm = nn.LSTM(input_size=1024,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            )
        self.transformer = nn.Transformer(d_model=128, batch_first=True)
        self.pre = nn.Linear(1024,128)
        self.node_choose = nn.Sequential(nn.Linear(128, 30522), nn.Softmax())
        self.node_reduce = nn.Sequential(nn.Linear(1024, 128), nn.Dropout())
        self.DIC_conv = GCNLayer(1024, 128)
        self.reduce = nn.Sequential(nn.Linear(256, 128), nn.Dropout())
        self.T_conv = GCNLayer(128, 128)
        self.S_conv = GCNLayer(128, 128)
        self.e = nn.Linear(128, 128)
        self.eps = 1e-12


    def forward(self, i, similarity_matrix_sub, A_sch, nodes_sch_embedding, sch_adj, T_adj, S_adj, nodes_sch_S_T, idx_list):

        # 准备空矩阵，为子图节点与边表示存储做准备
        node_target = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1]).to(self.device)
        T_sch_adj = torch.zeros_like(sch_adj).to(self.device)
        T_adj_sub = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1],
                                nodes_sch_embedding.shape[1]).to(self.device)
        S_adj_sub = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1],
                                nodes_sch_embedding.shape[1]).to(self.device)
        node_target[:, i] = 1

        block = 1

        # 选择当前时间步子图，服务于增量学习
        nodes_sch_time = torch.zeros_like(nodes_sch_embedding, device=self.device)
        nodes_sch_time[:, 0:i, :] = nodes_sch_embedding[:, 0:i, :].clone()
        # entry_loss = self.encoder(nodes_sch_time, node_id_all, node_target)
        nodes_embedding = self.pre(nodes_sch_embedding)
        outputs, (hn, cn) = self.Lstm(nodes_sch_embedding[:, 0:i, :])
        output = self.transformer(nodes_embedding[:, 0:i, :], nodes_embedding[:, 0:i, :])
        # output = self.pre(nodes_sch_time)
        step_loss = 0
        node_id = self.node_choose(output[:, -1, :])
        choose = torch.max(node_id, dim=1)
        # 增量学习每一步节点选择loss
        # targets = torch.zeros(nodes_sch_embedding.shape[0], 1).to(self.device)
        # targets[:, :] = idx_list[i]
        # targets = targets.to(torch.int64)
        probs = torch.gather(node_id, dim=-1, index=idx_list[i].unsqueeze(1)).squeeze()
        step_loss = -torch.log(probs + self.eps)
        # entry_loss = F.cross_entropy(node_id, node_target)
        # node_total_loss = entry_loss
        entry_loss = step_loss.mean()
        accuracy_gen = torch.sum(torch.eq(idx_list[i].unsqueeze(1), choose.indices)) / nodes_sch_embedding.shape[0]

        # 新事件节点加入后，事件节点时间信息聚合更新过程
        nodes_sch_time[:, 0:i + 1, :] = nodes_sch_embedding[:, 0:i + 1, :].clone()
        T_sch_adj[:, 0:i + 1, 0:i + 1] = sch_adj[:, 0:i + 1, 0:i + 1].clone()
        a = T_sch_adj[:, 0:i + 1, 0:i + 1]
        nodes_sch = nodes_sch_time[:, 0:i + 1, :]
        # nodes_sch = self.node_reduce(nodes_sch)

        nodes_sch_S = self.DIC_conv(T_sch_adj[:, 0:i + 1, 0:i + 1], nodes_sch)
        nodes_sch_S = torch.tanh(nodes_sch_S) + torch.sigmoid(nodes_sch_S)
        x = torch.matmul(nodes_sch_S, nodes_sch_S.transpose(1, 2)) / 128
        # 新事件节点加入后，事件节点空间信息聚合更新过程
        # 获取边关系子图
        T_adj_sub[:, 0:i + 1, 0:i + 1] = T_adj[0:i + 1, 0:i + 1].clone()
        S_adj_sub[:, 0:i + 1, 0:i + 1] = S_adj[0:i + 1, 0:i + 1].clone()
        nodes_sch_T = self.T_conv(T_adj_sub[:, 0:i + 1, 0:i + 1], nodes_sch_S) + nodes_sch_S
        nodes_sch_T = self.S_conv(S_adj_sub[:, 0:i + 1, 0:i + 1], nodes_sch_T) + nodes_sch_T
        nodes_sch_S_T = nodes_sch_T
        # nodes_sch_S_T = nodes_sch_S + nodes_sch_T
        # scores_sch = self.schemas_scores.compute_scores(nodes_sch_S)
        # standard_normal = Normal(0, 1)
        # 获取事件图总表示
        event_distribution = torch.mean(nodes_sch_S_T, dim=1)
        # New+Old事件节点拼接
        # for h in range(0,i):
        #     for z in range(0,i):
        #         # a = nodes_sch_S_T[:, h, :]
        #         # b = nodes_sch_S_T[:, z, :]
        #         trend = torch.cat((nodes_sch_S_T[:, h, :], nodes_sch_S_T[:, z, :]), dim=1)
        #         trend = self.reduce(trend)
        #         a = torch.cosine_similarity(trend, event_distribution, dim=-1)
        #         for b in range(0, similarity_matrix_sub.shape[0]):
        #             similarity_matrix_sub[b, h, z] = a[b]


        if i == 9:
            nodes_sch_seq = nodes_sch_S_T[:, 0:i, :] + nodes_sch_S_T[:, 0:i, :].clone()
        else:
            nodes_sch_seq = nodes_sch_S_T[:, 0:i, :] + nodes_sch_S_T[:, 1:i + 1, :].clone()

        # 计算随时间演绎相似度并作为边
        similarity = torch.cosine_similarity(self.e(nodes_sch_seq), event_distribution.unsqueeze(1), dim=-1)
        for j in range(0, i):
            for a in range(0, similarity_matrix_sub.shape[0]):
                similarity_matrix_sub[:, i, j] = similarity[:, j].clone()
                similarity_matrix_sub[:, j, i] = similarity[:, j].clone()
            # similarity_matrix_sub[:, j, j + 1] = similarity[:, j].clone()
            # similarity_matrix_sub[:, j + 1, j] = similarity[:, j].clone()

        return node_id, similarity_matrix_sub, nodes_sch_S_T, entry_loss, accuracy_gen

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class BaseSGNN(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(BaseSGNN, self).__init__()
        self.l = limit
        self.Bert = myBert()
        self.hidden_size = hidden_size

        self.gnn = GNN(self.hidden_size)

        self.linear_u_one = nn.Linear(hidden_size, int(0.5 * hidden_size), bias=True)
        self.linear_u_one2 = nn.Linear(int(0.5 * hidden_size), 1, bias=True)
        self.linear_u_two = nn.Linear(hidden_size, int(0.5 * hidden_size), bias=True)
        self.linear_u_two2 = nn.Linear(int(0.5 * hidden_size), 1, bias=True)

        self.multi = Parameter(torch.ones(3))

        self.MARGIN = 0.15
        self.loss_path_function = nn.BCELoss()

    def set_l(self, l):
        self.l = l

    def forward(self, nodes, A):
        hidden = self.gnn(A, nodes)
        scores = self.compute_scores(hidden)
        return scores, hidden

    def loss_node(self, output, target):
        output = output.view(5, -1)
        score_node = torch.mean(output, dim=1, keepdim=True)
        truth = score_node[target: target + 1, :].repeat(5, 1)
        loss = torch.sum((self.MARGIN + score_node - truth).clamp(min=0))
        return loss

    def accu_node(self, output, target):
        score_node = torch.mean(output.view(-1, 5), dim=0, keepdim=True)
        sorted, L = torch.sort(score_node, dim=-1, descending=True)
        target_index = torch.nonzero(L == target)
        rank = (4 - target_index[0, 1]) / 4
        return rank

    def loss_path(self, output, target):
        loss = self.loss_path_function(output, target)
        return loss

    def accu_path(self, output, target):
        p_truth = torch.nonzero(target == 1)
        p_false = torch.nonzero(target == 0)
        no_exist_count = p_false.shape[0]
        truth = output[p_truth[:, 0]].repeat(1, no_exist_count)
        other = output[p_false[:, 0]].T
        zeros = torch.zeros_like(truth)
        ones = torch.ones_like(truth)
        temp = torch.where(truth - other > 0, ones, zeros)
        temp1 = torch.sum(temp, dim=1) / no_exist_count
        accu = torch.mean(temp1, dim=0)
        return accu

    def compute_scores(self, hidden, metric='euclid'):
        # attention on input
        input_a = hidden[:, 0: self.l - 1, :]
        input_b = hidden[:, self.l - 1:, :]
        u_a = F.relu(self.linear_u_one(input_a))
        u_a2 = F.relu(self.linear_u_one2(u_a))
        u_b = F.relu(self.linear_u_two(input_b))
        u_b2 = F.relu(self.linear_u_two2(u_b))
        u_c = torch.add(u_a2, u_b2)
        weight = torch.exp(torch.tanh(u_c)).view(u_c.shape[0], -1)
        weight = (weight / torch.sum(weight, 1).view(-1, 1)).view(u_c.shape[0], -1, 1)
        weighted_input = torch.mul(input_a, weight)
        a = torch.sum(weighted_input, 1)
        b = input_b / (self.l - 1)
        b = b.view(b.shape[0], -1)
        if metric == 'dot':
            scores = metric_dot(a, b)
        elif metric == 'cosine':
            scores = metric_cosine(a, b)
        elif metric == 'euclid':
            scores = metric_euclid(a, b)
        elif metric == 'norm_euclid':
            scores = metric_norm_euclid(a, b)
        elif metric == 'manhattan':
            scores = metric_manhattan(a, b)
        elif metric == 'multi':
            scores = self.multi[0] * metric_euclid(a, b) + \
                     self.multi[1] * metric_dot(a, b) + \
                     self.multi[2] * metric_cosine(a, b)
        else:
            scores = metric_dot(a, b)
        return scores


class SchemasSGNN(BaseSGNN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(SchemasSGNN, self).__init__(limit, hidden_size)


class SkeletonSGNN(BaseSGNN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, limit, hidden_size):
        super(SkeletonSGNN, self).__init__(limit, hidden_size)


def metric_dot(v0, v1):
    return torch.sum(v0 * v1, 1).view(-1, 1)


def metric_cosine(v0, v1):
    return F.cosine_similarity(v0, v1).view(-1, 1)


def metric_euclid(v0, v1):
    return -torch.norm(v0 - v1, 2, 1).view(-1, 1)


def metric_norm_euclid(v0, v1):
    v0 = v0 / torch.norm(v0, 2, 1).view(-1, 1)
    v1 = v1 / torch.norm(v1, 2, 1).view(-1, 1)
    return -torch.norm(v0 - v1, 2, 1).view(-1, 1)


def metric_manhattan(v0, v1):
    return -torch.sum(torch.abs(v0 - v1), 1).view(-1, 1)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, node_features):
        # 使用邻接矩阵进行传播
        x = torch.matmul(adjacency_matrix, node_features)
        x = self.linear(x)
        x = F.relu(x)  # 激活函数可以根据任务调整

        return x


class GNN(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.2):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_ah = Parameter(torch.Tensor(self.hidden_size))

        self.dropout = nn.Dropout(dropout_p)
        self.reset_parameters()

    def GNNCell(self, A, hidden, w_ih, w_hh, b_ih, b_hh, b_ah):
        input = torch.matmul(A.transpose(1, 2), hidden)
        input = self.dropout(input)
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        hy = self.dropout(hy)
        return hy

    def forward(self, A, hidden):
        hidden1 = self.GNNCell(A, hidden, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        hidden2 = self.GNNCell(A, hidden1, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        return hidden2

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
