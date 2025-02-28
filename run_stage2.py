import os
import random
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torch_geometric.nn import TransformerConv, GraphConv, GCN2Conv, GCNConv, DenseGraphConv
from torch.utils.data import random_split,Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from observer import Runtime_Observer

class Liver_dataset(torch.utils.data.Dataset):
    def __init__(self, summery_path: str = 'data/summery_new.txt', feature_path: str = 'data/vision_features', fold = 0, mode: str = 'vis'):
        print("Dataset init ...")
        self.data_dict = {}  # all details of data
        self.data_list = []  # all uids of data
        self.data = []  # all value of data
        self.mode = mode
        summery = open(summery_path, 'r')
        titles = summery.readline().split()
        titles = [title.replace('_', ' ') for title in titles]
        df = pd.read_csv(f'data/vision_features_666/vision_features_{fold}.csv', header = 0, dtype={'uid': str})
        # 创建一个空字典来存储标签和对应的特征列表
        label_features = {}

        # 遍历DataFrame中的每一行
        for index, row in df.iterrows():
            label = str(row[0])  # 第一列是标签
            features = row[1:].tolist()  # 剩余列是特征
            label_features[label] = features
        
        count = 0
        for item in summery:
            count += 1
            single_data_list = item.split()
            temp_dict = {}
            temp = {}
            for i in range(len(single_data_list)):
                temp.update({titles[i]: single_data_list[i]})
            uid = temp.pop('uid')
            srcid = temp.pop('srcid')
            label = temp.pop('Outcome')
            _data = [float(x) for x in list(temp.values())]
            temp_dict.update({'srcid': srcid})
            temp_dict.update({'label': label})

            temp_dict.update({'features': _data})
            temp_dict.update({'vision_feature': label_features[uid]})
            # temp_dict.update({'source': temp})

            self.data_dict.update({uid: temp_dict})
            self.data_list.append(uid)
            self.data.append(_data)
        # Normalization

        print("Summery loaded --> Feature_num : %d  Data_num : %d" % (len(titles) - 3, count))
        summery.close()

    def __getitem__(self, index):
        uid = self.data_list[index]

        srcid = self.data_dict[uid]['srcid']
        text_feature = self.data_dict[uid]['features']
        
        vision_feature = self.data_dict[uid]['vision_feature']
        vision_feature = torch.Tensor(vision_feature)
        

        # usage of text data
        clinical = self.data_dict[uid]['features'][:-1781]
        clinical_tensor = torch.Tensor(clinical)
        clinical_tensor = torch.where(torch.isnan(clinical_tensor), torch.full_like(clinical_tensor, 0),
                                      clinical_tensor)
        # details
        basic_info_tensor = torch.Tensor(clinical[:21])
        blood_test_tensor = torch.Tensor(clinical[21:27])
        biochemical_tensor = torch.Tensor(clinical[27:44])
        fat_tensor = torch.Tensor(clinical[44:])
        # print(f"basic_info_tensor : {len(basic_info_tensor)}")
        # print(f"blood_test_tensor : {len(blood_test_tensor)}")
        # print(f"biochemical_tensor : {len(biochemical_tensor)}")
        # print(f"fat_tensor : {len(fat_tensor)}")
        # basic_info_tensor : 21
        # blood_test_tensor : 6
        # biochemical_tensor : 17
        # fat_tensor : 14


        radio = self.data_dict[uid]['features'][-1781:]
        radio_tensor = torch.Tensor(radio)
        radio_tensor = torch.where(torch.isnan(radio_tensor), torch.full_like(radio_tensor, 0), radio_tensor)

        label = int(self.data_dict[uid]['label'])
        label_tensor = torch.from_numpy(np.array(label)).long()
        
        if self.mode == 'vis':
            return clinical_tensor, vision_feature, label_tensor
        elif self.mode == 'base':
            return clinical_tensor, label_tensor
        else:
            return clinical_tensor, radio_tensor, label_tensor

    def __len__(self):
        return len(self.data_list)
    

class Graph_test(torch.nn.Module):
    def __init__(self, _debug = False, groups = [21, 6, 17, 14, 20], mode =  'random'):
        super().__init__()
        self.mode = mode
        self.debug = _debug
        self.total = sum(groups)
        self.groups = groups
        self.ratio = 0.2
        # self.Radio_autoencoder = nn.Sequential(
        #     nn.LayerNorm(1781),
        #     nn.Linear(1781, 20),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.2)
        # )
        self.vision_autoencoder = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 20),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2)
        )
        self.conv1 = TransformerConv(1, 1, heads=4, concat=True)
        self.conv2 = TransformerConv(4, 1, heads=1, concat=True)
        # self.conv1 = GraphConv(1, 4)
        # self.conv2 = GraphConv(4, 1)
        # self.conv1 = GCNConv(1, 4)
        # self.conv2 = GCNConv(4, 1)
        # self.conv1 = DenseGraphConv(1, 4)
        # self.conv2 = DenseGraphConv(4, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.total),
            nn.Linear(self.total, 2)
        )
        # Initialize weights for layers
        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print(f'layer {m} initialized')
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, cli_data, radio_data):
        radio_hidden = self.vision_autoencoder(radio_data)
        temp_features = torch.cat([cli_data, radio_hidden], dim=1)
        edges = []

        x = temp_features.view(-1, 1)

        if self.mode == 'fully_connected':
            for i in range(self.total):
                for j in range(self.total):
                    if i != j:
                        edges.append((i, j))

        elif self.mode == 'random':
            # full connect in one group
            offset = 0
            for num in self.groups:
                for i in range(num):
                    for j in range(num):
                        if i + offset != j + offset:
                            edges.append((i + offset, j + offset))
                offset += num
            # random connect between groups
            offset = 0
            count = 0
            for i in range(5):
                offset_2 = self.groups[i]
                for j in range(i + 1, 5):
                    # print(i, j)
                    for _i in range(self.groups[i]):
                        for _j in range(self.groups[j]):
                            # print(_i, _j)
                            if _i + offset != _j + offset_2 and random.random() < self.ratio:
                                edges.append((_i + offset, _j + offset_2))
                                # print(f"({_i + offset}, {_j + offset_2})")
                                count += 1
                                
                    offset_2 += self.groups[j]
                offset += self.groups[i]
            # print(self.ratio)
            # print(count)
            # print(len(edges))
            # exit(0)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = x.permute(1, 0)
        x = self.classifier(x)
        return x

    def _print(self, x):
        if self.debug:
            print(x)


device = 'cuda:7'

if __name__ == "__main__":
    seed = 666
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    torch.manual_seed(seed)
    for i in range(1):
        i = 4 
        dataset = Liver_dataset(fold=i)
        train_index, test_index = [[t1, t2] for t1, t2 in kf.split(dataset)][i]
        if not os.path.exists(f"debug/test{i}"):
            os.makedirs(f"debug/test{i}")
        observer = Runtime_Observer(log_dir=f"debug/test{i}", device=device, name="debug", seed=seed)
    
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        model = Graph_test(_debug=True, mode= 'random')
        optimizer = Adam(model.parameters(), weight_decay=1e-4)
        criterion = CrossEntropyLoss()
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                num_params += p.numel()
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")
    
        epochs = 200
        model = model.to(device)
        observer.log("start training\n")
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            observer.reset()
            model.train()
            train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
            running_loss = test_loss = 0.0
            for i, (info1, info2, label) in enumerate(train_bar):
                optimizer.zero_grad()
                info1, info2, label = info1.to(device), info2.to(device), label.to(device)
                outputs = model(info1, info2)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                # for name, param in model.named_parameters():
                #     print(name, param.grad)
                # exit()
                running_loss += loss.item() * info1.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            observer.log(f"Loss: {train_loss:.4f}\n")

            with torch.no_grad():
                model.eval()
                test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

                for i, (info1, info2, label) in enumerate(test_bar):
                    info1, info2, label = info1.to(device), info2.to(device), label.to(device)
                    outputs = model(info1, info2)
                    loss = criterion(outputs, label)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(outputs, dim=1)
                    confidence_scores = probabilities[range(len(predictions)), 1]
                    observer.update(predictions, label, confidence_scores)
                    test_loss += loss.item() * info1.size(0)
                test_loss = test_loss / len(train_loader.dataset)
                observer.log(f"Test Loss: {test_loss:.4f}\n")

            observer.record_loss(epoch, train_loss, test_loss)
            if observer.excute(epoch, model):
                print("Early stopping")
                break
        end_time = time.time()
        observer.log(f"\nRunning time: {end_time - start_time:.2f} seconds\n")
        observer.finish()
        


