import copy
import torch
from torch import nn
from models.embedding import myBert
from utils.util import my_norm


class Future(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, ontology):
        super(Future, self).__init__()
        self.edge_aware_gnn = EdgeAware()
        self.relation_enrichment = RelationEnrichment(ontology)
        self.temporal = Temporal()
        self.ontology = ontology
        self.myBert = myBert()
        self.map_e = nn.Linear(1024 * 2, 128, bias=True)
        self.map_v = nn.Linear(1024 * 2, 128, bias=True)
        self.map_a = nn.Linear(1024 * 2, 128, bias=True)
        self.map_r = nn.Linear(1024, 128, bias=True)

    def forward(self, graph):
        a_list = graph[0]
        r_list = graph[1]
        t_list = graph[2]
        events = graph[3]
        entities = graph[4]
        arguments = graph[5]
        relations = graph[6]
        e, v, a, r = self.pre_represent(
            events, entities, arguments, relations
        )
        e, v = self.edge_aware_gnn(
            e, v, a, r, a_list, r_list, t_list
        )
        temporal_p = self.temporal(e)
        return temporal_p

    def pre_represent(self, events, entities, arguments, relations):
        e = torch.zeros((len(events), 128)).to(self.device)
        v = torch.zeros((len(entities), 128)).to(self.device)
        a = torch.zeros((len(arguments), 128)).to(self.device)
        r = torch.zeros((len(relations), 128)).to(self.device)
        for i, event in enumerate(events):
            e_trigger = self.myBert(event["name"])
            e_type = self.myBert(event["@type"])
            e_all = self.map_e(torch.cat((e_trigger, e_type), dim=1))
            e[i: i + 1, :] = e_all
        for i, entity in enumerate(entities):
            v_trigger = self.myBert(entity["name"])
            v_type = self.myBert(entity["@type"])
            v_all = self.map_v(torch.cat((v_trigger, v_type), dim=1))
            v[i: i + 1, :] = v_all
        for i, argument in enumerate(arguments):
            a_trigger = self.myBert(argument["name"])
            a_type = self.myBert(argument["@type"])
            a_all = self.map_a(torch.cat((a_trigger, a_type), dim=1))
            a[i: i + 1, :] = a_all
        for i, relation in enumerate(relations):
            r_type = self.myBert(relation["@type"])
            r_all = self.map_r(r_type)
            r[i: i + 1, :] = r_all
        return e, v, a, r


class GraphConsolidation(Future):
    def forward(self, graph):
        a_list = graph[0]
        r_list = graph[1]
        t_list = graph[2]
        events = graph[3]
        entities = graph[4]
        arguments = graph[5]
        relations = graph[6]
        e, v, a, r = self.pre_represent(
            events, entities, arguments, relations
        )
        e, v = self.edge_aware_gnn(
            e, v, a, r, a_list, r_list, t_list
        )
        relations_p = self.relation_enrichment(e, v, a_list)
        return relations_p


class EdgeAware(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(EdgeAware, self).__init__()
        self.W_a = torch.nn.Parameter(torch.randn((256, 128), dtype=torch.float32), requires_grad=True)
        self.W_r = torch.nn.Parameter(torch.randn((256, 128), dtype=torch.float32), requires_grad=True)
        self.W_bfr = torch.nn.Parameter(torch.randn((128, 128), dtype=torch.float32), requires_grad=True)
        self.W_aft = torch.nn.Parameter(torch.randn((128, 128), dtype=torch.float32), requires_grad=True)
        self.MLP_t = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.MLP_a = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.MLP_r = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.MLP_e = nn.Sequential(
            nn.Linear(256, 128),
        )
        self.MLP_v = nn.Sequential(
            nn.Linear(256, 128),
        )
        self.GRU_e = nn.GRU(input_size=128, hidden_size=128, num_layers=2)
        self.GRU_v = nn.GRU(input_size=128, hidden_size=128, num_layers=2)
        self.layer_norm = nn.LayerNorm([128])
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, event_nodes, entity_nodes, argument_edges, relation_edges, a_list, r_list, t_list):
        a_list = copy.deepcopy(a_list)
        r_list = copy.deepcopy(r_list)
        t_list = copy.deepcopy(t_list)
        # 找到所有的事件节点的 id列表
        # 找到最后一个事件对应的实体 id列表
        # 找到其他事件的 id列表
        # 添加虚拟边
        event_count = len(event_nodes)
        for i in range(event_count - 1):
            t_list.append([i, event_count - 1])
        entity_count = len(entity_nodes)
        for j in range(entity_count - 1):
            r_list.append([j, entity_count - 1])
            relation_edges = torch.cat((relation_edges, torch.zeros((1, 128)).to(self.device)))
        events = torch.zeros((0, 128)).to(self.device)
        entities = torch.zeros((0, 128)).to(self.device)
        # 开始照着公式实现
        aggregate_message_a = torch.zeros((0, 128)).to(self.device)
        aggregate_message_r = torch.zeros((0, 128)).to(self.device)
        aggregate_message_t = torch.zeros((0, 128)).to(self.device)
        # 传递在参数边的信息
        for step1, arg in enumerate(a_list):
            e = event_nodes[arg[0]: arg[0] + 1, :]
            v = entity_nodes[arg[1]: arg[1] + 1, :]
            a = argument_edges[step1: step1 + 1, :]
            temp = torch.cat((e - v, a), dim=1)
            m_i_j = self.ReLU(temp @ self.W_a)
            a_i_j = self.MLP_a(e - v)
            a_m = a_i_j * m_i_j
            aggregate_message_a = torch.cat((aggregate_message_a, a_m))
        # 传递在关系边的信息
        for step2, rel in enumerate(r_list):
            v_j = entity_nodes[rel[0]: rel[0] + 1, :]
            v_k = entity_nodes[rel[1]: rel[1] + 1, :]
            r = relation_edges[step2: step2 + 1, :]
            m_j_k = self.ReLU(torch.cat((v_j - v_k, r), dim=1) @ self.W_r)
            a_j_k = self.MLP_r(v_j - v_k)
            r_m = a_j_k * m_j_k
            aggregate_message_r = torch.cat((aggregate_message_r, r_m))
        # 传递在时间边的信息
        for step3, tem in enumerate(t_list):
            e_i = event_nodes[tem[0]: tem[0] + 1, :]
            e_l = event_nodes[tem[1]: tem[1] + 1, :]
            t = e_i @ self.W_bfr - e_l @ self.W_aft
            m_i_l = self.ReLU(t)
            a_i_l = self.MLP_t(e_i - e_l)
            t_m = a_i_l * m_i_l
            aggregate_message_t = torch.cat((aggregate_message_t, t_m))
        # 更新事件节点的表示
        for i in range(event_nodes.shape[0]):
            e_n = torch.zeros((1, 128)).to(self.device)
            e_i = event_nodes[i: i + 1, :]
            for step4, tempor in enumerate(t_list):
                if tempor[0] == i or tempor[1] == i:
                    e_n = e_n + aggregate_message_t[step4: step4 + 1, :]
            for step5, argument in enumerate(a_list):
                if argument[0] == i:
                    e_n = e_n + aggregate_message_a[step5: step5 + 1, :]
            e_ner = self.layer_norm(e_n)
            e_i_n = torch.cat((e_i, e_ner))
            output, hn = self.GRU_e(e_i_n)
            events = torch.cat((events, self.MLP_e(output.reshape(1, -1))), dim=0)
        # 更新实体节点的表示
        for i in range(entity_nodes.shape[0]):
            v_n = torch.zeros((1, 128)).to(self.device)
            v_i = entity_nodes[i: i + 1, :]
            for step4, relation in enumerate(r_list):
                if relation[0] == i or relation[1] == i:
                    v_n = v_n + aggregate_message_r[step4: step4 + 1, :]
            for step5, argument in enumerate(a_list):
                if argument[1] == i:
                    v_n = v_n + aggregate_message_a[step5: step5 + 1, :]
            v_ner = self.layer_norm(v_n)
            v_i_n = torch.cat((v_i, v_ner))
            output, hn = self.GRU_v(v_i_n)
            entities = torch.cat((entities, self.MLP_v(output.reshape(1, -1))), dim=0)
        return [
            events, entities
        ]


class RelationEnrichment(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, ontology):
        super(RelationEnrichment, self).__init__()
        # Relation types: R + O(no relation)
        self.ontology = ontology
        self.MLP_r = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, len(ontology["relations"]) + 1),
            nn.ReLU()
        )
        self.MLP_non = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, e, v, a_list):
        new_event = len(e) - 1
        new_entity, old_entity = 0, 0
        for argument in a_list:
            if argument[0] == new_event:
                is_coref = False
                for argument1 in a_list:
                    if argument1[0] != new_event and argument1[1] == argument[1]:
                        is_coref = True
                if not is_coref:
                    new_entity = new_entity + 1
        old_entity = len(v) - new_entity
        relations_p = torch.zeros((new_entity, old_entity, len(self.ontology["relations"]) + 2),
                                  dtype=torch.float32)
        for j in range(old_entity, len(v)):
            for k in range(old_entity):
                r = torch.exp(self.MLP_r(v[j: j + 1, :] - v[k: k + 1, :]))
                r_sum = torch.sum(r)
                r_p = self.Sigmoid(r / r_sum)
                p_r_non = self.MLP_non(v[j: j + 1, :] - v[k: k + 1, :])
                r_p = torch.cat((r_p, p_r_non), dim=1)
                relations_p[j - old_entity][k] = r_p[0]
        return relations_p


class Temporal(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(Temporal, self).__init__()
        self.num_heads = 8
        self.map = nn.Linear(128 * self.num_heads, 128)
        self.GRU = nn.GRU(input_size=128, hidden_size=128, num_layers=2)
        self.qkv = nn.Linear(128, 128 * 3 * self.num_heads, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.W_q = torch.nn.Parameter(
            torch.randn((128, 128), dtype=torch.float32), requires_grad=True
        )
        self.W_k = torch.nn.Parameter(
            torch.randn((128, 128), dtype=torch.float32), requires_grad=True
        )
        self.layer_norm = nn.LayerNorm([128])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, e):
        l = len(e) - 1
        temporal_p = torch.zeros((1, l + 1), dtype=torch.float32)
        events = e[: l + 1, :]
        events = self.layer_norm(events)
        row, col = events.shape
        # 多头注意力
        qkv = self.qkv(events).reshape(row, 3, self.num_heads, col).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        content = (attn @ v).permute(1, 2, 0).reshape(row, col * self.num_heads)
        events = self.layer_norm(self.map(content) + events)
        e_now = events[l: l + 1, :]
        # 注意力的一部分
        q_1 = e_now @ self.W_q
        k_1 = events @ self.W_k
        e_score = (q_1 @ k_1.transpose(-2, -1)).T
        e_score = my_norm(e_score, 0)
        p_t = self.sigmoid(e_score).T
        temporal_p[0] = p_t[0]
        return temporal_p
