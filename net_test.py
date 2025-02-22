import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import models
from transformers import AutoTokenizer, AutoModel, AutoConfig

from data.dataset import Liver_dataset, Liver_normalization_dataset
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.model_selection import KFold



if __name__ =='__main__':

    from torch.utils.data import DataLoader
    import numpy as np
    device='cuda'
    seed = 42
    torch.manual_seed(seed)
    # dataset = Liver_normalization_dataset("./data/summery_new.txt", mode='radio_img_label')

    dataset = Liver_normalization_dataset("./data/summery_new.txt", mode='new')

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, test_index = [[t1, t2] for t1, t2 in kf.split(dataset)][4]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    from Net.networks import *
    from Net.cp_networks import *

    # model = VAE_CLNet().to(device)
    # model = Resnet_2D().to(device)
    model = CLNet_test(vision_encoder_selection=0, vision_dim = 512, use_attention= True).to(device)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    
    print("model parameters: " + str(num_params))
    

    model.eval()
        
    # for batch_idx, (batch_data) in enumerate(testDataLoader):
    #     if isinstance(batch_data, dict):
    #         inputs, inputs_2 = (
    #                 batch_data["image"].to(device),
    #                 batch_data["image_2"].to(device),
    #             )
    #     else:
    #         inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)
    #     print(inputs_2.shape)

    for batch_idx, (radio, img, label) in enumerate(testDataLoader):
        print(f"batch_idx: {batch_idx}  | radio shape: {radio.shape} | img shape: {img.shape}")
        # img = img[:,:,:,:64,:64]
        # print(img)
        img = img.to(device)

        radio = radio.to(device)
        label = label.to(device)


        # radio_feature, vision_feature, cli_feature, output = model(token, segment, mask, radio, img)
        logit, IB_loss, disen_loss, task_loss = model(radio, img, label)
        
        print(logit.shape)
        print(IB_loss)
        print(disen_loss)
        print(task_loss)
        # loss = nn.CrossEntropyLoss(output, label)

        
        exit()