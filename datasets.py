import json
import random

from torch.utils.data import Dataset
import torch
import copy


class Dataset1(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Dataset2(Dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, graph, ontology):
        self.ontology = ontology
        self.graph = [
            self.a_list, self.r_list, self.t_list,
            self.events, self.entities,
            self.arguments, self.relations, self.effe_path
        ] = create_full_graph(graph)

    def __getitem__(self, index):
        # index start from 0
        event_graph = self.create_graph(index)
        new_event = self.create_event(index + 1)
        return [event_graph, new_event]

    def __len__(self):
        return len(self.events) - 1

    def create_graph(self, index):
        # 根据当前索引构建含此索引的历史事件图
        events = self.events[0: index + 1]
        a_list = []
        r_list = []
        t_list = []
        entities = []
        arguments = []
        relations = []
        v_index = 0
        for a_id, arg in enumerate(self.a_list):
            if arg[0] <= index:
                arg1 = copy.deepcopy(self.arguments[a_id])
                if self.entities[arg[1]] in entities:
                    arg1["target"] = entities.index(self.entities[arg[1]])
                    arguments.append(arg1)
                    a_list.append([arg[0], arg1["target"]])
                else:
                    arg1["target"] = v_index
                    arguments.append(arg1)
                    entities.append(self.entities[arg[1]])
                    a_list.append([arg[0], v_index])
                    v_index += 1
        for r_id, rel in enumerate(self.relations):
            source, target = 0, 0
            find_s, find_t = False, False
            for v_id, ent in enumerate(entities):
                if rel["relationSubject"] == ent["@id"] and not find_s:
                    find_s = True
                    source = v_id
                elif rel["relations"]["relationObject"] == ent["@id"] and not find_s:
                    find_t = True
                    target = v_id
                if find_s and find_t:
                    break
            if find_s and find_t:
                rel = copy.deepcopy(rel)
                r_list.append([source, target])
                rel["source"] = source
                rel["target"] = target
                relations.append(rel)
        for tem in self.t_list:
            if tem[0] <= index and tem[1] <= index:
                t_list.append(tem)
        return [a_list, r_list, t_list, events, entities, arguments, relations]

    def create_event(self, index):
        # 根据当前索引构建此索引的事件图
        events = self.events[index: index + 1]
        a_list = []
        r_list = []
        entities = []
        arguments = []
        relations = []
        v_index = 0
        for a_id, arg in enumerate(self.a_list):
            if arg[0] == index:
                arg1 = copy.deepcopy(self.arguments[a_id])
                arg1["source"] = 0
                arg1["target"] = v_index
                arguments.append(arg1)
                entities.append(self.entities[arg[1]])
                a_list.append([0, v_index])
                v_index += 1
        for r_id, rel in enumerate(self.relations):
            source, target = 0, 0
            find_s, find_t = False, False
            for v_id, ent in enumerate(entities):
                if rel["relationSubject"] == ent["@id"] and not find_s:
                    find_s = True
                    source = v_id
                elif rel["relations"]["relationObject"] == ent["@id"] and not find_s:
                    find_t = True
                    target = v_id
                if find_s and find_t:
                    break
            if find_s and find_t:
                rel = copy.deepcopy(rel)
                r_list.append([source, target])
                rel["source"] = source
                rel["target"] = target
                relations.append(rel)
        return [a_list, r_list, events, entities, arguments, relations]


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
        if len(path) == 9:
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


def choose_four_events(graph):
    events = graph[3]
    entities = graph[4]
    arguments = graph[5]
    a_list = graph[0]
    four_events = []
    while len(four_events) < 4:
        i = random.randint(0, len(events) - 1)
        a, v, arg = [], [], []
        for j, a_l in enumerate(a_list):
            if a_l[0] == j:
                arguments[j]["source"] = 0
                arguments[j]["target"] = len(v)
                a.append([0, len(v)])
                arg.append(arguments[j])
                v.append(entities[a_l[1]])
        # [a_list, r_list, events, entities, arguments, relations]
        four_events.append([a, [], [events[i]], v, arg, []])
    return four_events
