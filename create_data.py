import pickle
import random

import torch
from datasets import Dataset1, Dataset2, create_full_graph, choose_four_events
from utils.util import find_all_json, load_schema, truth_guide_graph
from utils.ontology import read_ontology_excel
from transformers import BertTokenizer, BertModel
import json

# 获取IED Schema Learning Corpus
data_train = find_all_json("./data/Wiki_IED_split/train")
data_test = find_all_json("./data/Wiki_IED_split/test")

# data_train = find_all_json("./data/LDC_schema_corpus_ce_split/train")
# data_test = find_all_json("./data/LDC_schema_corpus_ce_split/test")

train_dataset = Dataset1(data_train)
test_dataset = Dataset1(data_test)

ontology = read_ontology_excel("./data/kairos-ontology.xlsx")

instance2schema = {
    "suicide_ied_train": "General-IED",
    "suicide_ied_test": "General-IED",
    "suicide_ied_dev": "General-IED",
    "wiki_drone_strikes_train": "drone-strikes-IED",
    "wiki_drone_strikes_test": "drone-strikes-IED",
    "wiki_drone_strikes_dev": "drone-strikes-IED",
    "wiki_ied_bombings_train": "General-IED",
    "wiki_ied_bombings_test": "General-IED",
    "wiki_ied_bombings_dev": "General-IED",
    "wiki_mass_car_bombings_train": "car-IED",
    "wiki_mass_car_bombings_test": "car-IED",
    "wiki_mass_car_bombings_dev": "car-IED",
}

schemas = load_schema("./data/RESIN_schema/resin-schemalib.json")
# 数据集制造选择：train/test
mode = "train"
graph_dataset = train_dataset if mode == "train" else test_dataset
graphs = []
graph_i = 11
if mode == "train":
    data_len = 1
if mode == "test":
    data_len = 1


def create_full_graph(graph):
    a_list = []
    r_list = []
    t_list = []
    events = []
    entities = []
    arguments = []
    relations = []
    # 抽取图中所有实体的类型 entity types,
    for entity in graph["schemas"][0]["entities"]:
        entity["@type"] = entity["entityTypes"].split("/")[-1]
        entities.append(entity)
    # 抽取图中所有事件的类型 event types,
    for e_id, event in enumerate(graph["schemas"][0]["steps"]):
        participants = event["participants"]
        event["@type"] = event["@type"].split("/")[-1]
        events.append(event)
        for role in participants:
            role["@type"] = role["role"].split("/")[-1]
            v_id = 0
            for id_, v in enumerate(entities):
                if role["values"][0]["entity"] == v["@id"]:
                    v_id = id_
            a_list.append([e_id, v_id])
            role["source"] = e_id
            role["target"] = v_id
            arguments.append(role)
    for r in graph["schemas"][0]["entityRelations"]:
        r["@type"] = r["relations"]["relationPredicate"].split("/")[-1]
        s, t = 0, 0
        find_s, find_t = False, False
        for v_id, v in enumerate(entities):
            if r["relationSubject"] == v["@id"] and not find_s:
                find_s = True
                s = v_id
            elif r["relations"]["relationObject"] == v["@id"] and not find_t:
                find_t = True
                t = v_id
            if find_s and find_t:
                break
        r_list.append([s, t])
        r["source"] = s
        r["target"] = t
        relations.append(r)
    for t in graph["schemas"][0]["order"]:
        source, target = 0, 0
        find_s, find_t = False, False
        for e_id, e in enumerate(events):
            if t["before"] == e["@id"] and not find_s:
                find_s = True
                source = e_id
            elif t["after"] == e["@id"] and not find_t:
                find_t = True
                target = e_id
            if find_s and find_t:
                break
        t_list.append([source, target])

    # 深度遍历图
    def edges_to_adjacency_list(edges):
        adjacency_list = {}
        for edge in edges:
            head, tail = edge  # 假设每个表示是一个包含头节点和尾节点的元组
            if head in adjacency_list:
                adjacency_list[head].append(tail)
            else:
                adjacency_list[head] = [tail]
        return adjacency_list
    # 转换为邻接表表示
    adjacency_list = edges_to_adjacency_list(t_list)

    def dfs(graph, node, path, paths):
        # 将当前节点添加到路径中
        path.append(node)
        # 如果当前节点是终点（没有出边），则将路径添加到路径列表中
        if node not in graph:
            paths.append(path[:])  # 使用path[:]创建路径的副本
        else:
            # 递归遍历所有相邻节点
            for neighbor in graph[node]:
                dfs(graph, neighbor, path, paths)
        # 从路径中移除当前节点，以便回溯
        path.pop()
    def find_all_paths(graph):
        paths = []
        for node in graph:
            dfs(graph, node, [], paths)
        return paths
    all_paths = find_all_paths(adjacency_list)
    effe_path = []
    for path in all_paths:
        # 链的长度
        if len(path) == graph_i:
            if len(effe_path) != 0:
                count = sum([1 for x, y in zip(path, effe_path[-1]) if x == y])
                if count <= 2:
                    effe_path.append(path)
            else:
                effe_path.append(path)
        # if len(path) == 9:
        #     if len(effe_path) != 0:
        #         # count = sum([1 for x, y in zip(path[0], effe_path[-1][0]) if x==y])
        #         if path[0] != effe_path[-1][0]:
        #             effe_path.append(path)
        #     else:
        #         effe_path.append(path)
    # print()

    return [a_list, r_list, t_list, events, entities, arguments, relations, effe_path]


tokenizer = BertTokenizer.from_pretrained(r".\bert-large-uncased")

result_list = []
test_list = []
train_list = []
# 遍历train、test中的每一张图
for i, data in enumerate(graph_dataset):
    # train
    if i in [13, 27, 38, 46, 102, 154, 173, 176, 177, 182, 200, 201, 208, 220, 226, 268, 271, 274, 278, 314, 315, 319, 332, 342]:
        continue
    # if i >= 200:
    #     continue
    # test
    # if i in [37, 44]:
    #     continue
    schema = schemas[instance2schema[data["name"]].lower()]
    datas = create_full_graph(data)
    eff_path_list = datas[7]
    events_list = datas[3]
    j = 0
    if i == 3:
        print(1)
    for eff_path in eff_path_list:
        if j == 130:
            continue

        if i == 5 and j == 76:
            print(1)
        j += 1
        result = [events_list[i]['name'] for i in eff_path]
        # train_list.append(result)
        # test_list.append(result)
        if j <= 20 and j <= len(eff_path_list) * 0.1:
            test_list.append(result)
        else:
            train_list.append(result)
    print(i)

# def get_result(eff_path_list, events_list, eff_path):

print()
# 示例字符列表


# 打开文件并将字符列表写入文件
# with open("./data/new/{}/{}{}.data".format(mode, mode, graph_i), "w") as file:
with open('./data/new/{}/{}node{}.json'.format('test', 'test', graph_i), 'w') as file:
    json.dump(test_list, file)

with open('./data/new/{}/{}node{}.json'.format('train', 'train', graph_i), 'w') as file:
    json.dump(train_list, file)


    # 省去
    # single_dataset = Dataset2(data, ontology)
    # if i < len(graph_dataset) - 1:
    #     false_graph = create_full_graph(graph_dataset[i + 1])
    # else:
    #     false_graph = create_full_graph(graph_dataset[0])
    # for j, [event_graph, new_event] in enumerate(single_dataset):
    #     g_events = event_graph[3]
    #     g_entities = event_graph[4]
    #     n_events = new_event[2]
    #     n_entities = new_event[3]
    #
    #     if not n_events:
    #         break
    #     if len(g_events) > 100:
    #         break
    #     if len(g_events) < 16:
    #         continue
    #
    #     candidate_events = choose_four_events(false_graph)
    #     target = random.randint(0, 4)
    #     candidate_events.insert(target, new_event)
    #
    #     temporal_orders, create_relations = truth_guide_graph(event_graph, new_event, schema)
    #     truth_paths = torch.zeros((5, 1, len(g_events) + 1), dtype=torch.float32)
    #     truth_paths[:, 0, len(g_events)] = 1.
    #     create_temporal = False
    #     temporal_orders.reverse()
    #     for t in temporal_orders:
    #         if not create_temporal or t[1] - t[0] <= 30:
    #             create_temporal = True
    #             truth_paths[target][0][len(g_events)] = 0.
    #             truth_paths[target][0][t[0]] = 1.
    #     graph = [event_graph, candidate_events, target, truth_paths]
    #     graphs.append(graph)
    #
    # if len(graphs) >= data_len:
    #     with open("./data/new/{}/{}{}.data".format(mode, mode, graph_i), "wb") as f:
    #         pickle.dump(graphs, f)
    #         print("{}{}.data saved".format(mode, graph_i))
    #     graph_i += 1
    #     graphs = []
    # else:
    #     continue
