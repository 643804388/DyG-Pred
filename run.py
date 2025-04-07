import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
torch.cuda.current_device()
import os
# from utils.util import add_graph, convert_graph, my_sub_graph, random_sort
from models.future import Future
from models.sgnn3 import SGNN
# from utils.ontology import read_ontology_excel
# import networkx as nx
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import torch.nn.functional as F
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_list = []
test_list = []
writer = SummaryWriter(log_dir="./logs")
now_time = str(time.strftime("%Y-year-%m-month-%d-day-%H-hour-%M-minute-%S-second"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(r'./data/new/train/train3.json', 'r') as file:
    train_data = json.load(file)
with open(r'./data/new/test/test3.json', 'r') as file:
    test_data = json.load(file)

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
node = 10
ins = node
sch = 16
batch_size = 16
sgnn = SGNN(ins, sch, device, num_nodes=node, dropout=0.3, gcn_bool=True, addaptadj=True, seq_length=128,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=64, end_channels=128,
                 layers=2, embed_dim=node, dropout_ingc=0.5, eta=1, gamma=0.001,
                 m=0.9, batch_size=64, dilation_exponential_=1).to(device)

optimizer = torch.optim.Adam(sgnn.parameters(), lr=3e-4, weight_decay=1e-5)
nparams = sum([p.nelement() for p in sgnn.parameters()])
# 允许多少比例的事件进入
prob = 0.2

# 生成时序序列矩阵
S_adj = torch.zeros(ins, ins)
T_adj = torch.zeros(ins, ins)
for i in range(0, node-1):
    S_adj[i][i + 1] = 1
    S_adj[i + 1][i] = 1
    T_adj[i][i + 1] = 1


# 单独链检测(数据格式)
# test_data = [['moving', 'attack', 'identified', 'attacks', 'arrested', 'released', 'flown', 'visited', 'summit', 'protests', 'battle'],['attacks', 'attack', 'arrested', 'released', 'flown', 'visited', 'summit', 'wounding', 'wounded', 'casualty', 'wounded']]
# test_data = [['attacks', 'attack', 'arrested', 'released', 'flown', 'visited', 'summit', 'wounding', 'citing', 'killed', 'wounded']]
# test_data = [['moving', 'attack', 'identified', 'attacks', 'arrested', 'released', 'flown', 'visited', 'summit', 'protests', 'battle']]

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
iter_wrapper_train = (lambda x: tqdm(x, total=len(train_loader), ncols=80))
iter_wrapper_test = (lambda x: tqdm(x, total=len(test_loader), ncols=80))
step = 0
train = 1
test = 1


for epoch in range(0, 200):
    if train == 1:
        # 训练
        # sgnn.load_state_dict(torch.load('models/output/example_model{}.pth'.format(27)))
        sgnn.train()
        sgnn.zero_grad()
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        entry_loss_all = 0
        current_loss = 0
        current_acc = 0
        # for i, li in tqdm(enumerate(train_data)):
        for s, li in iter_wrapper_train(enumerate(train_loader)):

            Results = sgnn(s, batch_size, li, S_adj.unsqueeze(0), li, S_adj.unsqueeze(0), S_adj, T_adj, node)
            (step_loss_all, graph_loss,  node_choose_loss, accuracy_gen, structure_similarity_score, attribute_similarity_score, similarity_score, edge_diff, accuracy_list) = Results
            # loss1 = sgnn.loss_node(score_i, target)
            # loss2 = sgnn.loss_path(score_i, path_target)
            # loss = step_loss_all
            # loss = step_loss_all
            loss = step_loss_all + 0.1 * graph_loss + 1 * node_choose_loss
            sgnn.zero_grad()
            # with torch.autograd.detect_anomaly():
                # loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer.step()
            current_loss += step_loss_all.item()
            current_acc += accuracy_gen.item()
            entry_loss_all += node_choose_loss.item()
            # entry_loss_all += node_choose_loss
            # accu1 = sgnn.accu_node(score_i, target)
            # accu2 = sgnn.accu_path(score_i, path_target)
            # print("loss1:", float(loss1), "loss2:", float(loss2), "accu1:", float(accu1), "accu2:", float(accu2))
            # print("loss:", float(loss), "accu:", float(accuracy_gen), )
            writer.add_scalars("{}/train/loss".format(now_time), {"node_loss": loss}, step)
            writer.add_scalars("{}/train/loss".format(now_time), {"node_accuracy": accuracy_gen}, step)
            writer.add_scalars("{}/train/loss".format(now_time), {"graph_loss": graph_loss}, step)
            writer.add_scalars("{}/train/loss".format(now_time), {"node_choose_loss": node_choose_loss}, step)
            step += 1
            # writer.add_scalars("{}/train/accu".format(now_time), {"node": accu1}, step)
            # writer.add_scalars("{}/train/accu".format(now_time), {"path": accu2}, step)
        s = s+1
        current_loss_avg = current_loss / s
        current_acc_avg = current_acc / s
        entry_loss_avg = entry_loss_all / s
        # if epoch >= 2 and epoch % 2 == 0:
        #     torch.save(sgnn.state_dict(), 'models/output/example_modelmcnc{}.pth'.format(epoch))
        print("epoch:", epoch, "loss:", current_loss_avg, "entry_loss:", entry_loss_avg, "acc:", current_acc_avg)
        writer.add_scalars("{}/train/loss".format(now_time), {"current_loss_avg": current_loss_avg}, epoch)
        writer.add_scalars("{}/train/loss".format(now_time), {"current_acc_avg": current_acc_avg}, epoch)
        writer.add_scalars("{}/train/loss".format(now_time), {"entry_loss_avg": entry_loss_avg}, epoch)
    if test == 1:
        # 测试
        # sgnn.load_state_dict(torch.load('models/output/example_model11{}.pth'.format(30)))
        sgnn.eval()
        entry_loss_all_test = 0
        current_loss_test = 0
        current_acc_test = 0
        structure_similarity_score_test = 0
        attribute_similarity_score_test = 0
        similarity_score_test = 0
        edge_diff_test = 0
        accuracy_all = []
        # for i, li in tqdm(enumerate(train_data)):
        for s, li in iter_wrapper_test(enumerate(test_loader)):
            Results = sgnn(s, batch_size, li, S_adj.unsqueeze(0), li, S_adj.unsqueeze(0), S_adj, T_adj, node)
            (step_loss_all, graph_loss,  node_choose_loss, accuracy_gen, structure_similarity_score, attribute_similarity_score, similarity_score, edge_diff, accuracy_list) = Results
            # loss1 = sgnn.loss_node(score_i, target)
            # loss2 = sgnn.loss_path(score_i, path_target)
            accuracy_all.append(accuracy_list)
            loss = step_loss_all
            current_loss_test += loss.item()
            current_acc_test += accuracy_gen.item()
            entry_loss_all_test += node_choose_loss.item()
            attribute_similarity_score_test += attribute_similarity_score.item()
            structure_similarity_score_test += structure_similarity_score.item()
            similarity_score_test += similarity_score.item()
            edge_diff_test += edge_diff.item()

            # entry_loss_all_test += node_choose_loss
            # attribute_similarity_score_test += attribute_similarity_score
            # structure_similarity_score_test += structure_similarity_score
            # similarity_score_test += similarity_score
            # edge_diff_test += edge_diff

            # accu1 = sgnn.accu_node(score_i, target)
            # accu2 = sgnn.accu_path(score_i, path_target)
            # print("loss1:", float(loss1), "loss2:", float(loss2), "accu1:", float(accu1), "accu2:", float(accu2))
            # print("loss:", float(loss), "accu:", float(accuracy_gen), )
            writer.add_scalars("{}/test/loss".format(now_time), {"node_loss": loss}, step)
            writer.add_scalars("{}/test/loss".format(now_time), {"node_accuracy": accuracy_gen}, step)
            writer.add_scalars("{}/test/loss".format(now_time), {"graph_loss": graph_loss}, step)
            writer.add_scalars("{}/test/loss".format(now_time), {"node_choose_loss": node_choose_loss}, step)
            step += 1
            # writer.add_scalars("{}/train/accu".format(now_time), {"node": accu1}, step)
            # writer.add_scalars("{}/train/accu".format(now_time), {"path": accu2}, step)
        s = s+1
        accuracy_all = torch.stack(accuracy_all)
        accuracy_sum = torch.sum(accuracy_all, dim=0) / s

        current_loss_avg = current_loss_test / s
        current_acc_avg = current_acc_test / s
        entry_loss_avg = entry_loss_all_test / s
        attribute_similarity_score_avg = attribute_similarity_score_test / s
        structure_similarity_score_avg = structure_similarity_score_test / s
        similarity_score_avg = similarity_score_test / s
        edge_diff_avg = edge_diff_test / s


        print("epoch:", epoch, "loss:", current_loss_avg, "entry_loss_test:", entry_loss_avg, "acc_test:", current_acc_avg,
              "attribute_similarity:", attribute_similarity_score_avg, "structure_similarity:", structure_similarity_score_avg,
              "similarity:", similarity_score_avg, "edge_diff:", edge_diff_avg, 'accuracy_sum:', accuracy_sum)
        writer.add_scalars("{}/test/loss".format(now_time), {"current_loss_avg": current_loss_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"current_acc_avg": current_acc_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"entry_loss_avg": entry_loss_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"attribute_similarity_score_avg": attribute_similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"structure_similarity_score_avg": structure_similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"similarity_score_avg": similarity_score_avg}, epoch)
        writer.add_scalars("{}/test/loss".format(now_time), {"edge_diff": edge_diff_avg}, epoch)
