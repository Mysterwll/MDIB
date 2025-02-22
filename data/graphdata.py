import pandas as pd
import numpy as np
import torch.nn.functional as F
# import dgl
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度。

    参数：
    x (np.ndarray): 第一个向量，类型为 NumPy 数组。
    y (np.ndarray): 第二个向量，类型为 NumPy 数组。

    返回：
    float: 余弦相似度，值范围从 -1 到 1。
    """
    
    # 输入验证：检查是否输入的是一维 NumPy 数组
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Both inputs must be numpy arrays.")
    
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional arrays.")

    # 计算余弦相似度
    dot_product = np.dot(x, y)  # 计算点积
    norm_x = np.linalg.norm(x)  # 计算第一个向量的范数
    norm_y = np.linalg.norm(y)  # 计算第二个向量的范数
    
    # 如果任何一个向量的范数为0，余弦相似度无法定义
    if norm_x == 0 or norm_y == 0:
        raise ValueError("One or both input vectors are zero vectors, which cannot have a cosine similarity.")

    cosine_similarity = dot_product / (norm_x * norm_y)  # 计算余弦相似度

    return cosine_similarity

def adj_matrix(patient_info, mode = 1):
    label = patient_info['label'].tolist()
    clinical = patient_info['clinical']
    clinical = (clinical - np.min(clinical, axis=0)) / (np.max(clinical , axis=0) - np.min(clinical, axis=0))
    vision = patient_info['vision']

    # mean_vals = np.mean(clinical, axis=0)
    # std_vals = np.std(clinical, axis=0)
    # clinical = (clinical - mean_vals) / std_vals

    if mode ==1:
        x = vision
        edge = clinical
    elif mode == 2:
        x = np.concatenate((vision, clinical), axis=1)
        edge = clinical
    elif mode == 3:
        x = vision
        edge = np.concatenate((vision, clinical), axis=1)
    elif mode == 4:
        x = np.concatenate((vision, clinical), axis=1)
        edge = np.concatenate((vision, clinical), axis=1)

    edge_list=[]
    edge_wight=[]
    n_sample = len(label)
    adj = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            adj[i,j] = cosine_similarity_matrix(edge[i],edge[j])
            if adj[i,j] > 0.5 and i!=j:
                # print(i,j)
                edge_list.append([i,j])
                edge_wight.append(adj[i,j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # 转换为COO格式的边索引
    edge_attr = torch.tensor(edge_wight, dtype=torch.float)


    
    x = torch.from_numpy(x.astype(np.float32))


    return x, edge_index, edge_attr


# def graph_bulider(all_data):

#     # save the labels
#     label = all_data['label']
#     # norm_label = (final_data['OS_Month']-np.min(final_data['OS_Month']))/(np.max(final_data['OS_Month'])-np.min(final_data['OS_Month']))
#     label = torch.from_numpy(label)
    
#     adj_sh, edge_list_sh, edge_wight_sh = adj_matrix(all_data)
#     print("the number of nodes in this graph:",len(label))
#     print("the number of edges in this graph:",len(edge_list_sh))
#     print("Number of average degree: ",len(edge_list_sh)/len(label) )
    
#     # build graph struture data
#     g_sh = dgl.DGLGraph()
#     g_sh.add_nodes(len(label))
#     # add nodes
#     # node_feature = (all_data.iloc[:, 15:]-all_data.iloc[:, 15:].min())/(all_data.iloc[:, 15:].max()- all_data.iloc[:, 15:].min())
#     node_feature_sh = all_data['vision']
#     # print(node_feature)
    
#     g_sh.ndata['h'] = torch.from_numpy(node_feature_sh).float()
#     g_sh.ndata['label'] = label
#     g_sh.ndata
#     # g.adj = adj
#     # add edges
#     src, dst = tuple(zip(*edge_list_sh))
#     g_sh.add_edges(src, dst)
#     # add edge weight
#     edge_wight_sh = np.array(edge_wight_sh)
#     g_sh.edata['w'] = torch.from_numpy(edge_wight_sh).float()
#     return adj_sh, g_sh

def get_pinfo(cli_file, vision_file):
    # file_path = "summery_new.xlsx"
    patient_info = {}
    
    data = pd.read_excel(cli_file, skiprows=1, header=None)
    v_data = pd.read_csv(vision_file, skiprows=1, header=None)

    uid = data.iloc[:, 0].to_numpy()  # 第一列：uid
    outcome = data.iloc[:, 1].to_numpy()  # 第二列：outcome
    clinical = data.iloc[:, 2:].to_numpy()  # 从第三列开始为临床信息

    uid_ = v_data.iloc[:, 0].to_numpy()
    vision = v_data.iloc[:, 1:].to_numpy()

    patient_info['uid']= uid
    patient_info['label']= outcome
    patient_info['clinical'] = clinical
    patient_info['vision'] = vision

    assert np.array_equal(uid, uid_) , "Please match Patient between two excels!"
    
    # adj, edge_list, edge_wight = adj_matrix(patient_info)
    # print(adj.shape)
    # print(len(edge_list))

    # graph_bulider(patient_info)
    return patient_info
    




        

