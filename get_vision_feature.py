import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import models
from transformers import AutoTokenizer, AutoModel, AutoConfig

from data.dataset import Liver_dataset, Liver_normalization_dataset
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.model_selection import KFold
from tqdm import tqdm
import sys
import pandas as pd
from Net.networks import *
from Net.cp_networks import *
from torch.utils.data import DataLoader
import numpy as np

def get_vision_feature(model_path):

    device='cuda'
    dataset = Liver_normalization_dataset("./data/summery_new.txt", mode='radio_img_label_test')

    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # model = CLNet().to(device)
    model = CTEncoder(embedding_dim = 512, projection_head= False).to(device)
    
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    
    print("model parameters: " + str(num_params))
    
    model.eval()
        
    with torch.no_grad():
        test_bar = tqdm(testDataLoader, leave=True, file=sys.stdout)
        data = []  
        for batch_idx, (uid, radio, img, label) in enumerate(test_bar):
            img = img.to(device)
            radio = radio.to(device)
            label = label.to(device)
            
            vision_feature = model(img)
                        
            for i in range(vision_feature.shape[0]):
                
                uid_str = uid[i] if isinstance(uid[i], str) else uid[i].item()
                data.append([uid_str] + vision_feature[i].tolist())
                
                # data.append([srcid[i].item()] + vision_feature[i].tolist())
            
            # print(srcid)
            # print(vision_feature.shape)    
    
    # 将数据转换为pandas DataFrame
    df = pd.DataFrame(data, columns=['uid'] + [f'feature_{i}' for i in range(vision_feature.shape[1])])

    # 保存到CSV文件
    csv_path = 'vision_features.csv' 
    df.to_csv(csv_path, index=False)

    print(f'Data saved to {csv_path}')
    
if __name__ =='__main__':

    model_path = "./logs/SimCLR/Resnet50/checkpoints/best_model.pth"
    get_vision_feature(model_path)