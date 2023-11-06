import re
from torch.utils.data import Dataset
import numpy as ny
import torch
from torch.nn import functional as F
import numpy as np


def match_parentheses(expression):
    pattern = r'(\([^()]+\))|([^()]+)'
    matches = re.finditer(pattern, expression)
    result = []

    for match in matches:
        if match.group(1):
            result.append(re.sub(r', ', '', match.group(1)[1:-1]))  # 括号内的内容
        else:
            result.append(re.sub(r', ', '', match.group(2)))  # 括号外的内容

    return result


def generate_edge_emb(node_l):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    symbol_dict = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8, 'rand': 9, 'pad': 10}
    # symbol = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (len(symbol_dict), dim)), dtype=torch.float, requires_grad=True, device=device))

    edge_emb = []
    for node in node_l:
        if node in symbol_dict:
            edge_emb.append(symbol_dict[node])
        else:
            edge_emb.append(symbol_dict[node])
    edge_emb = torch.stack(edge_emb, dim=0)

    return edge_emb


# def pad_collate(x):
#     """
#     performing padding for graph
#
#     max_num_nodes = max(graph.num_nodes for graph in batch)
#     for graph in batch:
#         num_padding = max_num_nodes - graph.num_nodes
#         graph.x = F.pad(graph.x, (0, 0, 0, num_padding))
#         graph.edge_index = F.pad(graph.edge_index, (0, num_padding), value=graph.num_nodes)
#     """
#     """
#     each batch format should be:
#     {
#         'equation1': {'edge_index': tensor(), 'node_list': tensor()},
#         'target': {'edge_index': tensor(), 'node_list': tensor()}
#     }
#     """
#     symbol_dict = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8, 'rand': 9, 'pad': 10}
#     max_num_nodes = max(len(graph['equation1']['node_list']) for graph in x)
#
#     output_edge, output_node = [[], []], []
#     for i, graph in enumerate(x):
#         edge, node = graph['equation1']['edge_index'], graph['equation1']['node_list']
#         num_padding = max_num_nodes - len(node)
#
#         node_index = []
#         for n in node:
#             n = n.strip()
#             if n in symbol_dict:
#                 node_index.append(symbol_dict[n])
#             else:
#                 node_index.append(symbol_dict['rand'])
#
#         node_index = torch.tensor([node_index])
#         node_index = F.pad(node_index, (0, num_padding), value=symbol_dict['pad']).tolist()
#         edge = F.pad(edge, (0, num_padding), value=max_num_nodes).tolist()
#
#         output_node.append(node_index)
#         output_edge[0] += edge[0]
#         output_edge[1] += edge[1]
#
#     return torch.tensor(output_node).squeeze(1), torch.tensor(output_edge)


def pad_collate(x):
    return x
