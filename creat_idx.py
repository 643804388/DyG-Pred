import torch

from datasets import Dataset1, Dataset2, create_full_graph, choose_four_events
from utils.util import find_all_json, load_schema, truth_guide_graph
from utils.ontology import read_ontology_excel
from transformers import BertTokenizer, BertModel
import json
from transformers import BertTokenizer, BertModel
from transformers import logging
import torch
from torch import nn
graph_i = 0

with open('data/new/test/test3.json', 'r') as file:
    test_data = json.load(file)
with open('data/new/train/train3.json', 'r') as file:
    train_data = json.load(file)

class myBert(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(myBert, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained(r"F:\Plot_Reasoning\bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(r"F:\动态图生成\bert-large-uncased")
        self.model = BertModel.from_pretrained(r"F:\动态图生成\bert-large-uncased").to(self.device)

    def forward(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # 将tokens字符串映射到其词汇索引vocabulary indices。
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens[1]
tokenizer = BertTokenizer.from_pretrained(r"F:\动态图生成\bert-large-uncased")
for train_idx in train_data:
    batch_list = []
    schadj_list = []
    idx_list = []
    node_embedding_list = []
    for batch in train_idx:
        if len(idx_list) <= 9:
            idx = tokenizer(batch)

            idx_list.append(idx['data']['input_idx'][1])
        else:
            idx = tokenizer(batch)
            target = idx
    print()


with open('./data/new/{}/{}{}.json'.format('test', 'test', graph_i), 'w') as file:
    json.dump(test_list, file)

with open('./data/new/{}/{}{}.json'.format('train', 'train', graph_i), 'w') as file:
    json.dump(train_list, file)