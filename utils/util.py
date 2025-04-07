import copy
import json

import torch


def find_all_json(path):
    """
    给出训练集对应的路径，读取该路径下所有的json文件，并将其中的json合并在一起
    :param path:
    :return:
    """
    from pathlib import Path

    glob_generator = Path(path).glob("*.json")
    data = []
    for element in glob_generator:
        name = str(element).split("\\")[-1].replace(".json", "")
        with open(element, 'r', encoding="utf-8") as f:
            line_list = f.read().split('\n')
            if ('') in line_list:
                line_list.remove('')
            for line in line_list:
                json_data = json.loads(line)
                json_data["name"] = name
                data.append(json_data)
    return data


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        lib = json.load(f)

    schemas = lib["schemas"]
    res = {}
    for sc in schemas:
        a_l, r_l, t_l, events, entities, arguments, relations = [], [], [], [], [], [], []
        for step in sc["steps"]:
            e = {
                "@id": step["@id"],
                "name": step["name"],
                "@type": step["@type"].split("/")[-1],
                "comment": step["comment"]
            }
            events.append(e)
            for role in step["participants"]:
                v_type = [t.split("/")[-1] for t in role["entityTypes"]]
                v = {
                    "@id": role["@id"],
                    "@type": ",".join(v_type),
                    "name": role["name"].split("/")[-1]
                }
                entities.append(v)
                a = {
                    "@id": role["@id"],
                    "@type": role["role"].split("/")[-1],
                    "name": role["name"].split("/")[-1],
                    "source": len(events) - 1,
                    "target": len(entities) - 1
                }
                arguments.append(a)
                a_l.append([len(events) - 1, len(entities) - 1])
        for order in sc["order"]:
            if type(order["before"]) != list:
                order["before"] = [order["before"]]
            if type(order["after"]) != list:
                order["after"] = [order["after"]]
            for ei in order["before"]:
                ei_id = find_event_index(events, ei)
                for el in order["after"]:
                    el_id = find_event_index(events, el)
                    t_l.append([ei_id, el_id])
        for rel in sc["entityRelations"]:
            r_s = rel["relationSubject"]
            r_s_id = find_entity_index(entities, r_s)
            for r1 in rel["relations"]:
                if type(r1["relationObject"]) != list:
                    r1["relationObject"] = [r1["relationObject"]]
                for r_o in r1["relationObject"]:
                    r_o_id = find_entity_index(entities, r_o)
                    r = {
                        "@id": r1["@id"],
                        "name": r1["name"],
                        "@type": r1["relationPredicate"].split("/")[-1],
                        "source": r_s_id,
                        "target": r_o_id
                    }
                    relations.append(r)
                    r_l.append([r_s_id, r_o_id])
        graph = [a_l, r_l, t_l, events, entities, arguments, relations]
        res[sc["@id"].split("/")[-1].lower()] = graph
    return res


def find_event_index(events, e_id):
    for i, e in enumerate(events):
        if e["@id"] == e_id:
            return i
    return None


def find_entity_index(entities, v_id):
    for i, v in enumerate(entities):
        if v["@id"] == v_id:
            return i
    return None


def truth_guide_graph(event_graph, new_event, schema):
    g_events = event_graph[3]
    n_events = new_event[2]
    # 1. 找到新加入事件在schema中对应的节点
    # 1.1 找到在schema中与新加入事件类型相同的节点，若只有一个，则直接匹配
    new_event_type = n_events[0]["@type"]
    schema_events = schema[3]
    match_list = []
    for i, e in enumerate(schema_events):
        if e["@type"] == new_event_type:
            match_list.append(i)
    if len(match_list) != 1:
        # 1.2 若有多个，则看schema中指向该节点的节点在当前图instance中是否存在，若有不存在的则直接排除
        for c in match_list[::-1]:
            is_exist = False
            for t in schema[2]:
                if t[1] == c:
                    t0_type = schema_events[t[0]]["@type"]
                    for e in g_events:
                        if e["@type"] == t0_type:
                            is_exist = True
                            break
                    if is_exist:
                        break
            if not is_exist:
                match_list.remove(c)
        # 1.3 若存在的仍有多个，则看那些存在的节点的时间顺序，越接近当前事件的为正确的节点
        if len(match_list) != 1:
            e_max = 0
            for m in match_list:
                if m > e_max:
                    e_max = m
            match_list = [e_max]
    e_matched = match_list[0]
    # 2. 找到对应的节点后，找对应的参数和实体
    # 2.1 根据当前事件的参数与实体在schema中查找与之相应的参数和实体
    now_v2s_v = {}
    for i, a in enumerate(schema[0]):
        if a[0] == e_matched:
            for a_n in new_event[4]:
                if a_n["@type"] == schema[5][i]["@type"]:
                    now_v2s_v[a_n["target"]] = a[1]
    # print("now_v2s_v:", now_v2s_v)

    # 3. 构建实体之间关系的真值 (包括合并共指，体现为 Physical.SameAs.SameAs)
    create_relations = []
    coref_arguments = []
    # 3.1 查找对应的实体与哪些实体存在关系，并查找那些实体对应的事件类型
    for k1 in now_v2s_v:
        v1 = now_v2s_v[k1]
        # [[[source, target], event_id, r_id], ...]
        s_t_e = []
        for r_id, r1 in enumerate(schema[1]):
            if r1[0] == v1 or r1[1] == v1:
                v2 = r1[1] if r1[0] == v1 else r1[0]
                e2 = 0
                for a1 in schema[0]:
                    if a1[1] == v2:
                        e2 = a1[0]
                s_t_e.append([r1, e2, r_id])
        for r1, e, r_id in s_t_e[::-1]:
            # 3.2 查找那些事件类型在schema中是否在当前事件对应的节点之前，实则保留，反之排除
            if e > e_matched:
                s_t_e.remove([r1, e, r_id])
                continue
            # 3.3 若那些事件类型在当前图中不存在，则排除，反之保留，并在当前图找到与之对应的节点
            e_type = schema[3][e]["@type"]
            is_find = False
            for i_e in g_events:
                if i_e["@type"] == e_type:
                    is_find = True
            if not is_find:
                s_t_e.remove([r1, e, r_id])
                continue
        # print(s_t_e)
        # 3.4 与那些事件类型对应的instance中的节点的参数实体建立相应的实体间关系，构建实体关系的真值
        for r, e, r_id in s_t_e:
            e_type = schema[3][e]["@type"]
            r_type = schema[6][r_id]["@type"]
            v2 = r[1] if r[0] == v1 else r[0]
            a2 = schema[0].index([e, v2])
            for e_id, i_e in enumerate(g_events):
                if i_e["@type"] == e_type:
                    # 把 relation 另一边的实体对应的参数找到，找到它在 instance 中的位置
                    for a1_id, a1 in enumerate(event_graph[0]):
                        if a1[0] == e_id:
                            if schema[5][a2]["@type"] == event_graph[-2][a1_id]["@type"]:
                                k2 = a1[1]
                                # 3.5 合并共指体现为 Physical.SameAs.SameAs，构建合并共指的真值
                                if r_type == "Physical.SameAs.SameAs":
                                    coref_arguments.append([len(event_graph[-3]) + k1, k2])
                                create_relations.append([len(event_graph[-3]) + k1, k2, r_type])
    # print("create_relations:", create_relations)
    # print("coref_arguments:", coref_arguments)

    # 4. 构建事件时间关系边的真值
    temporal_orders = []
    # 4.1 查看schema中对应的节点之前 (一跳) 的事件类型节点
    for t in schema[2]:
        if t[1] == e_matched:
            # 4.2 查找instance中与之对应的节点，构建事件时间边的真值
            i_e_type = schema[3][t[0]]["@type"]
            for e_id, i_e in enumerate(g_events):
                if i_e["@type"] == i_e_type:
                    temporal_orders.append([e_id, len(g_events)])
    # 4.3 若没找到，考虑二跳的情况，重复以上操作   tip: 只会有一条事件时间边
    # if len(temporal_orders) == 0:
    #     for t1 in schema[2]:
    #         is_find = False
    #         for t2 in schema[2]:
    #             if t1[1] == t2[0] and t2[1] == e_matched:
    #                 i_e_type = schema[3][t1[0]]["@type"]
    #                 for e_id, i_e in enumerate(g_events):
    #                     if i_e["@type"] == i_e_type:
    #                         temporal_orders.append([e_id, len(g_events)])
    #                         is_find = True
    #                         break
    #             if is_find:
    #                 break
    #         if is_find:
    #             break
    # print("temporal_orders:", len(temporal_orders))
    return temporal_orders, create_relations


def add_graph(new_event, event_graph):
    event_graph = copy.deepcopy(event_graph)
    e_len = len(event_graph[3])
    v_len = len(event_graph[4])
    for a in new_event[0]:
        event_graph[0].append([e_len + a[0], v_len + a[1]])
    for r in new_event[1]:
        event_graph[1].append([v_len + r[0], v_len + r[1]])
    for e in new_event[2]:
        event_graph[3].append(e)
    for v in new_event[3]:
        event_graph[4].append(v)
    for a in new_event[4]:
        a["source"] = e_len + a["source"]
        a["target"] = v_len + a["target"]
        event_graph[5].append(a)
    for r in new_event[5]:
        r["source"] = v_len + r["source"]
        r["target"] = v_len + r["target"]
        event_graph[6].append(r)
    return event_graph


def convert_graph(graph):
    # [a_list, r_list, t_list, events, entities, arguments, relations]
    a_list = graph[0]
    t_list = graph[2]
    events = graph[3]
    entities = graph[4]
    arguments = graph[5]
    g = {
        "events": [],
        "tempor": t_list
    }
    for i, event in enumerate(graph[3]):
        e = {
            "name": event["name"],
            "@type": event["@type"],
            "args": []
        }
        for j, li in enumerate(a_list):
            if li[0] == i:
                arg = {
                    "role": arguments[j]["@type"],
                    "name": arguments[j]["name"],
                    "entityType": entities[li[1]]["@type"]
                }
                e["args"].append(arg)
        g["events"].append(e)
    return g


def my_sub_graph(G, c_id, limit):
    sub = [c_id]
    k = 0
    while k < len(sub) <= limit:
        predecessors = list(G.predecessors(sub[k]))
        successors = list(G.successors(sub[k]))
        for predecessor in predecessors:
            if int(predecessor) not in sub:
                sub.append(int(predecessor))
        for successor in successors:
            if int(successor) not in sub:
                sub.append(int(successor))
        k += 1
    w = 0
    while len(sub) < limit:
        if w not in sub:
            sub.append(w)
        w += 1
    G_sub = G.subgraph(sub[0: limit])
    return G_sub


def random_sort(a, b):
    import random
    import numpy as np
    i = random.randint(0, 50)
    random.seed(i)
    a_list = torch2list(a)
    b_list = torch2list(b)
    index = [j for j in range(len(a_list))]
    random.shuffle(index)
    index = np.array(index)
    a_list = np.array(a_list)
    b_list = np.array(b_list)
    a_list = a_list[index]
    b_list = b_list[index]
    a = torch.tensor(a_list, dtype=torch.float32).view(1, -1)
    b = torch.tensor(b_list, dtype=torch.int64).view(1, -1)
    return a, b


def torch2list(tensor):
    ndndarray = tensor.numpy()
    li = list(ndndarray.reshape(len(ndndarray[0])))
    return li


def my_norm(x, dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_mean = torch.mean(x, dim=dim)
    if x.view(-1).shape[0] != 1:
        x_std = torch.std(x, dim=dim)
    else:
        x_std = torch.ones_like(x, device=device)
    y = (x - x_mean) / x_std
    return y
