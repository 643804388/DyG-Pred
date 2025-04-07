import math
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.embedding import myBert
from torch.nn import init
import numbers
import matplotlib.pyplot as plt
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SGNN(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, sch_hidden_size, ske_hidden_size, device, num_nodes, dropout=0.3, gcn_bool=True, addaptadj=True, seq_length=128,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=64, end_channels=128,
                 layers=2, embed_dim=10, dropout_ingc=0.5, eta=1, gamma=0.001,
                 m=0.9, batch_size=64, dilation_exponential_=1):
        super(SGNN, self).__init__()
        self.node = num_nodes
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
        self.x = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.LayerNorm([128]))
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
        self.choose = nn.Sequential(nn.Linear(in_features=116, out_features=30522),
                                    nn.Softmax(),
                                    )
        self.eps = 1e-12


        self.relu = nn.Sequential(nn.ReLU())
        self.schemas_S_T = SchemasSGNN(128, sch_hidden_size)
        self.gate_Fusion_1 = gatedFusion_1(128, device)
        self.graph_learn = Graph_learn(node_dim=128, heads=4, head_dim=8, nodes=self.node,
                                   eta=1, gamma=0.0001, dropout=0.5)
        self.dropout = dropout

        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv_s = nn.ModuleList()
        self.gconv_d = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.seq_length = seq_length
        kernel_size = 7

        dilation_exponential = dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        rf_size_i = 1
        new_dilation = 1
        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                # rf_size_j = 7, 19, 43, 91, 187
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))
            self.gate_convs.append(
                dilated_inception(residual_channels, dilation_channels, dilation_factor=new_dilation))

            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            if self.gcn_bool:
                self.gconv_s.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))
                self.gconv_d.append(gcn_module(dilation_channels, residual_channels, dropout, support_len=1, order=2))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                           elementwise_affine=True))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                           elementwise_affine=True))
            new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_x = nn.Conv2d(in_channels=32,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)
        self.idx = torch.arange(self.nodes).to(device)



    def set_l(self, l):
        self.schemas_sgnn.set_l(l)
        self.skeleton_sgnn.set_l(l)

    def forward(self, s, batch_size, nodes_sch, A_sch, nodes_ske, A_ske, S_adj, T_adj, node):
        # 获取图节点表示与静态图邻接矩阵
        batch_list = []
        schadj_list = []
        idx_list = []
        embedding_batch_list = []
        for batch in nodes_sch:
            idx_batch_list = []
            if len(embedding_batch_list) <= self.node-1:
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
        # 数据放入cuda
        nodes_sch_embedding, A_sch, sch_adj = nodes_sch_embedding.to(self.device), A_sch.to(self.device), sch_adj.to(self.device)
        S_adj, T_adj = S_adj.to(self.device), T_adj.to(self.device)
        similarity_matrix_list = []
        nodes_sch_S_T_list = []
        node_id_all = []
        node_choose_loss = 0.

        nodes_Static = nodes_sch_embedding = self.x(nodes_sch_embedding)
        nodes_sch_S_T = nodes_sch_embedding[:, 0:1, :]
        similarity_matrix_sub = torch.zeros_like(sch_adj)
        S_adj_loss = torch.zeros_like(sch_adj)
        accuracy_list = []
        # 事件预测+动态图重构，输入每一时间步加入的节点，输出重构之后的子图
        for i in range(1, nodes_sch_embedding.shape[1]):
            # 准备空矩阵，为子图节点与边表示存储做准备
            node_target = torch.zeros(nodes_sch_embedding.shape[0], nodes_sch_embedding.shape[1]).to(self.device)

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
        #     # similarity_all = (corr_similarity + normalized_tensor) / 2
        #     similarity_all = normalized_tensor
        #
        #     # 将二维张量转换为NumPy数组
        #     array = np.array(similarity_all.to('cpu').detach())
        #
        #     # 使用Matplotlib绘制图形
        #     plt.imshow(array, cmap='viridis', interpolation='nearest')
        #     plt.colorbar()  # 添加颜色条
        #     # plt.show()
        #     filepath = './view/STDyn-sgnn3/instanceview{}'.format(s)
        #     if not os.path.isdir(filepath):
        #         # 创建文件夹
        #         os.mkdir(filepath)
        #     plt.savefig('./view/STDyn-sgnn3/instanceview{}/plot{}.jpg'.format(s, t), format='jpg')
        #     plt.close()
        #
        # sch_array = np.array(sch_adj.to('cpu').detach()).squeeze(0)
        # plt.imshow(sch_array, cmap='viridis', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # # plt.show()
        # filepath = './view/STDyn-sgnn3/instanceview{}'.format(s)
        # if not os.path.isdir(filepath):
        #     # 创建文件夹
        #     os.mkdir(filepath)
        # plt.savefig('./view/STDyn-sgnn3/instanceview{}/sch{}.jpg'.format(s, s), format='jpg')
        # plt.close()

        S_adj_ex = S_adj.repeat(similarity_matrix_sub.shape[0], 1, 1)
        graph_loss = F.mse_loss(S_adj_ex, similarity_matrix_sub)
        accuracy_list = torch.stack(accuracy_list)
        # 邻接图评分
        # 邻接矩阵归一化
        # 最小-最大归一化
        # similarity_matrix_sub = similarity_all
        similarity_matrix_min_max_normalization = min_max_normalization(similarity_matrix_sub)

        median = similarity_matrix_min_max_normalization.median().item()
        graph_matrix_edge = torch.where(similarity_matrix_min_max_normalization > median, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
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
        edge_diff = abs(count.sum() - S_count.sum()) / nodes_Static.shape[0]

        nodes_S_T = nodes_sch_S_T_list[-1]
        # nodes_Static = self.x(nodes_sch_embedding)
        # 图融合
        nodevec_fusion = self.gate_Fusion_1(nodes_sch_S_T.shape[0], nodes_Static, nodes_sch_S_T) + nodes_Static
        adj = self.graph_learn(nodevec_fusion, sch_adj, nodes_Static, similarity_matrix_sub, batch_size)
        adj_d, adj_s, node_embed, Graph_loss = adj
        gl_loss = Graph_loss
        static_adj = adj_s
        dy_adj = adj_d
        static_adj_min_max_normalization = min_max_normalization(static_adj)
        st_median = static_adj_min_max_normalization.median().item()

        st_graph_matrix_edge = torch.where(static_adj_min_max_normalization >= st_median, torch.tensor(1).to(self.device),
                                           torch.tensor(0).to(self.device))
        # 可视化
        # 融合矩阵可视化
        # dy_adj = (adj_d + similarity_matrix_sub) / 2
        # dy_median = dy_adj.median().item()
        # dy_graph_matrix_edge = torch.where(dy_adj > dy_median, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
        #
        # sch_dy = np.array(dy_adj.to('cpu').detach()).squeeze(0)
        # plt.imshow(sch_dy, cmap='viridis', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # # plt.show()
        # filepath = './view/STDyn-sgnn3/instanceview{}'.format(s)
        # if not os.path.isdir(filepath):
        #     # 创建文件夹
        #     os.mkdir(filepath)
        # plt.savefig('./view/STDyn-sgnn3/instanceview{}/sch_dy{}.jpg'.format(s, s), format='jpg')
        # plt.close()
        #
        # sch_st = np.array(static_adj.to('cpu').detach()).squeeze(0)
        # plt.imshow(sch_st, cmap='viridis', interpolation='nearest')
        # plt.colorbar()  # 添加颜色条
        # # plt.show()
        # filepath = './view/STDyn-sgnn3/instanceview{}'.format(s)
        # if not os.path.isdir(filepath):
        #     # 创建文件夹
        #     os.mkdir(filepath)
        # plt.savefig('./view/STDyn-sgnn3/instanceview{}/sch_st{}.jpg'.format(s, s), format='jpg')
        # plt.close()

        # 获取融合图
        # S_T_Graph_nodes = torch.concat([nodes_S_T, nodes_Static], dim=-1)
        # S_T_Graph_nodes = self.reduce(S_T_Graph_nodes)
        x1 = nodes_Static.unsqueeze(1)
        skip = self.skip0(F.dropout(x1, self.dropout, training=self.training))
        x = self.start_conv(x1)
        # WaveNet layers
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)

            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_bool:
                x_s = self.gconv_s[i](x, static_adj)
                x_d = self.gconv_d[i](x, dy_adj)
                x = x_s + x_d
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        x = self.end_conv_x(x).squeeze(1)
        # skip = self.skipE(x) + skip
        # x = F.relu(skip)
        #
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        # x = x



        # q_s = self.q_S(nodes_S_T)
        # k_s = self.k_S(nodes_S_T)
        # v_s = self.v_S(S_T_Graph_nodes)
        # attn1 = (q_s @ k_s.transpose(-2, -1))
        # attn1 = attn1.softmax(dim=-1)
        # hidden_S = self.layer_norm_S(attn1 @ v_s + nodes_S_T)
        # # 融合图与静态图作交叉注意力
        # q_t = self.q_T(nodes_Static)
        # k_t = self.k_T(nodes_Static)
        # v_t = self.v_T(S_T_Graph_nodes)
        # attn2 = (q_t @ k_t.transpose(-2, -1))
        # attn2 = attn2.softmax(dim=-1)
        # hidden_T = self.layer_norm_T(attn2 @ v_t + nodes_Static)
        # z =  torch.sigmoid(torch.add(hidden_S, hidden_T))
        # H = torch.add(torch.mul(z, hidden_S), torch.mul(1 - z, hidden_T))
        P_dist = self.choose(torch.mean(x, dim=1))
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
        self.pre = nn.Linear(1024, 128)
        self.node_choose = nn.Sequential(nn.Linear(128, 30522), nn.Softmax())
        self.node_reduce = nn.Sequential(nn.Linear(1024, 128), nn.Dropout())
        self.DIC_conv = GCNLayer(128, 128)
        self.DIC_conv_norm = nn.LayerNorm(128)
        self.reduce = nn.Sequential(nn.Linear(256, 128), nn.Dropout())
        self.T_conv = GCNLayer(128, 128)
        self.T_conv_norm = nn.LayerNorm(128)
        self.S_conv = GCNLayer(128, 128)
        self.S_conv_norm = nn.LayerNorm(128)
        self.e = nn.Linear(128, 128)
        self.eps = 1e-12
        self.skip_norm = nn.LayerNorm(128)
    def forward(self, i, similarity_matrix_sub, A_sch, nodes_sch_embedding, sch_adj, T_adj, S_adj, nodes_sch_S_T,
                idx_list):

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

        # outputs, (hn, cn) = self.Lstm(nodes_sch_embedding[:, 0:i, :])
        output = self.transformer(nodes_sch_embedding[:, 0:i, :], nodes_sch_embedding[:, 0:i, :])
        # output = self.pre(nodes_sch_time)
        step_loss = 0
        node_id = self.node_choose(output[:, -1, :])
        # 增量学习每一步节点选择loss
        # targets = torch.zeros(nodes_sch_embedding.shape[0], 1).to(self.device)
        # targets[:, :] = idx_list[i]
        # targets = targets.to(torch.int64)
        probs = torch.gather(node_id, dim=-1, index=idx_list[i].unsqueeze(1)).squeeze()
        step_loss = -torch.log(probs + self.eps)
        # entry_loss = F.cross_entropy(node_id, node_target)
        # node_total_loss = entry_loss
        entry_loss = step_loss.mean()
        choose = torch.max(node_id, dim=1)

        accuracy_gen = torch.sum(torch.eq(idx_list[i], choose.indices)) / nodes_sch_embedding.shape[0]
        # 新事件节点加入后，事件节点时间信息聚合更新过程
        nodes_sch_time[:, 0:i + 1, :] = nodes_sch_embedding[:, 0:i + 1, :].clone()
        T_sch_adj[:, 0:i + 1, 0:i + 1] = sch_adj[:, 0:i + 1, 0:i + 1].clone()
        a = T_sch_adj[:, 0:i + 1, 0:i + 1]
        nodes_sch = nodes_sch_time[:, 0:i + 1, :]
        # nodes_sch = self.node_reduce(nodes_sch)

        nodes_sch_S = self.DIC_conv(T_sch_adj[:, 0:i + 1, 0:i + 1], nodes_sch)
        nodes_sch_S = torch.tanh(nodes_sch_S) + torch.sigmoid(nodes_sch_S)
        nodes_sch_S = self.DIC_conv_norm(nodes_sch_S)
        # 新事件节点加入后，事件节点空间信息聚合更新过程
        # 获取边关系子图
        T_adj_sub[:, 0:i + 1, 0:i + 1] = T_adj[0:i + 1, 0:i + 1].clone()
        S_adj_sub[:, 0:i + 1, 0:i + 1] = S_adj[0:i + 1, 0:i + 1].clone()
        nodes_sch_T = self.T_conv(T_adj_sub[:, 0:i + 1, 0:i + 1], nodes_sch_S) + nodes_sch_S
        nodes_sch_T = self.T_conv_norm(nodes_sch_T)
        nodes_sch_T = self.S_conv(S_adj_sub[:, 0:i + 1, 0:i + 1], nodes_sch_T) + nodes_sch_T
        nodes_sch_T = self.S_conv_norm(nodes_sch_T)
        # nodes_sch_T = self.S_conv(S_adj_sub[:, 0:i + 1, 0:i + 1], nodes_sch_S) + nodes_sch_S
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


class Graph_learn(nn.Module):
    def __init__(self, node_dim, heads, head_dim, nodes=207, eta=1,
                 gamma=0.001, dropout=0.5, n_clusters=5):
        super(Graph_learn, self).__init__()

        self.D = heads * head_dim  # node_dim #
        self.heads = heads
        self.dropout = dropout
        self.eta = eta
        self.gamma = gamma

        self.head_dim = head_dim
        self.node_dim = node_dim
        self.nodes = nodes

        self.query = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.value = fc_layer(in_channels=node_dim, out_channels=self.D, need_layer_norm=False)
        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)

        self.bn = nn.LayerNorm(node_dim)

        self.w = nn.Parameter(torch.zeros(size=(nodes, node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.attn_static = nn.LayerNorm(nodes)
        self.skip_norm = nn.LayerNorm(nodes)
        self.attn_norm = nn.LayerNorm(nodes)
        self.linear_norm = nn.LayerNorm(nodes)
        self.attn_linear = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear.data, gain=1.414)
        self.attn_linear_1 = nn.Parameter(torch.zeros(size=(nodes, nodes)))
        nn.init.xavier_uniform_(self.attn_linear_1.data, gain=1.414)
        self.static_inf_norm = nn.LayerNorm(nodes)
        self.attn_norm_1 = nn.LayerNorm(nodes)
        self.attn_norm_2 = nn.LayerNorm(nodes)

    def forward(self, nodevec_fusion, nodevec_s, node_input, nodevec_dy, batch_size=64, ):
        batch_size, nodes, node_dim = nodevec_fusion.shape[0], self.nodes, self.node_dim
        node_orginal = nodevec_s
        # Static Graph Structure Learning
        adj_static = self.static_graph(node_orginal)

        nodevec_fusion = self.bn(nodevec_fusion)

        # Inductive bias
        # static_graph_inf = self.static_inf_norm(torch.mm(nodevec_dy, nodevec_dy.transpose(1, 0)))

        # residual connection in Dynamic relationship construction
        nodevec1_1 = torch.einsum('bnd, nl -> bnl', nodevec_fusion, self.w) + nodevec_fusion
        skip_atten = torch.einsum('bnd,bdm->bnm', nodevec1_1, nodevec1_1.transpose(-1, -2))
        skip_atten = self.skip_norm(skip_atten)

        # Multi-Head Adjacent mechanism
        nodevec_fusion = nodevec_fusion.unsqueeze(1).transpose(1, -1)
        query = self.query(nodevec_fusion)
        key = self.key(nodevec_fusion)
        # value = self.value(nodevec_fusion)
        key = key.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes)
        query = query.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes).transpose(-1, -2)
        attention = torch.einsum('bhnd, bhdu-> bhnu', query, key)
        attention /= (self.head_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = self.mlp(attention) + attention
        adj_bf = self.attn_norm(torch.sum(attention, dim=1)) + skip_atten

        # feedforward neural network
        adj_af = F.relu(torch.einsum('bnm, ml->bnl', self.linear_norm(adj_bf), self.attn_linear))
        adj_af = torch.einsum('bnm, ml -> bnl', adj_af, self.attn_linear_1)

        # add & norm
        dy_adj_inf = self.attn_norm_1(adj_af + adj_bf + nodevec_dy)
        # dy_adj_inf = self.attn_norm_1(adj_af + adj_bf + nodevec_dy)
        dy_adj_inf = F.dropout(dy_adj_inf, self.dropout, training=self.training)

        # add Inductive bias
        # static_graph_inf = static_graph_inf.unsqueeze(0).repeat(batch_size, 1, 1)
        # dy_adj = self.attn_norm_2(dy_adj_inf + static_graph_inf)
        dy_adj = self.attn_norm_2(dy_adj_inf)

        # The final inferred dynamic graph structure
        adj_dynamic = F.softmax(F.relu(dy_adj), dim=2)
        # adj_static = adj_static.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_static = adj_static
        # Graph Structure Learning Loss
        gl_loss = None
        if self.training:
            gl_loss = self.graph_loss_orginal(node_input, adj_static, self.eta, self.gamma)
        return adj_dynamic, adj_static, node_orginal, gl_loss,

    def static_graph(self, nodevec):
        resolution_static = torch.bmm(nodevec, nodevec.permute(0, 2, 1))
        resolution_static = F.softmax(F.relu(self.attn_static(resolution_static)), dim=1)
        return resolution_static

    def graph_loss_orginal(self, input, adj, eta=1, gamma=0.001):
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        dist_loss = torch.sum(dist_loss, dim=(1, 2))
        f_norm = torch.pow(torch.norm(adj, dim=(1, 2)), 2)
        gl_loss = dist_loss + gamma * f_norm
        return gl_loss
class gcn_module(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_module, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)
class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data, gain=1.414)

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))\
                     # + self.layer_norm(self.linear(input).transpose(1, -1))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w])) \
                     # + self.linear(input).transpose(1, -1)
        return result.transpose(1, -1)


class gatedFusion_1(nn.Module):
    def __init__(self, dim, device):
        super(gatedFusion_1, self).__init__()
        self.device = device
        self.dim = dim
        self.w = nn.Linear(in_features=dim, out_features=dim)
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)
        self.norm = nn.LayerNorm(dim)
        self.re_norm = nn.LayerNorm(dim)

        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, batch_size, nodevec, time_node):

        if batch_size == 1 and len(time_node.shape) < 3:
            time_node = time_node.unsqueeze(0)

        nodevec = self.norm(nodevec)
        node_res = self.w(nodevec) + nodevec
        # node_res = batch_size, nodes, dim
        # node_res = node_res.unsqueeze(0).repeat(batch_size, 1, 1)

        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        # z = batch_size, nodes, dim
        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec))
        h = torch.tanh(self.w_h(time_node) + r * (self.w_u(nodevec)))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res


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
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # (residual_channels, num_nodes, self.seq_length - rf_size_j + 1) 这是第一个参数
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x