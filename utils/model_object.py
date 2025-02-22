from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from Net.loss_functions import *
from Net.api import *
from Net.networks import *
from Net.cp_networks import *
from Net.graph_net import *
from Net.cmib import *
from Net.mpgsurv import *

models = {
    'CLNet': {
        'Name': 'CLNet',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'radio_img_label',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'w':0.5
        }
    },
    'CLNet_0': {
        'Name': 'CLNet_0_3',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'radio_img_label',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 2,
           'vision_dim':2048,
           'w':0.5
        }
    },
    'CLNet_1': {
        'Name': 'CLNet_1',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'all_model',
        'Model': CLNet_1,
        'Optimizer': Adam,
        'Loss': joint_loss_1,
        'Run': run_all,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 2,
           'vision_dim':2048,
           'w':0.5
        }
    },
    'CLNet_2': {
        'Name': 'CLNet_2',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'new',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 0,
           'vision_dim':512,
           'w':0.5,
           'use_attention' : True
        }
    },
    'CLNet_3': {
        'Name': 'CLNet_3',
        'Data': './data/summery_new.txt',
        'Batch': 4,
        'Lr': 0.0001,
        'Epoch': 200,
        'Dataset_mode': 'new',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 0,
           'vision_dim':512,
           'w':0.5,
           'use_attention' : True
        }
    },
    'CLNet_4': {
        'Name': 'CLNet_4',
        'Data': './data/summery_new.txt',
        'Batch': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'Dataset_mode': 'new',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 0,
           'vision_dim':512,
           'w':0.5,
           'use_attention' : True
        }
    },
    'CLNet_5': {
        'Name': 'CLNet_5',
        'Data': './data/summery_new.txt',
        'Batch': 16,
        'Lr': 0.0001,
        'Epoch': 200,
        'Dataset_mode': 'new',
        'Model': CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 1,
        'Input_args': {
           'vision_encoder_selection': 0,
           'vision_dim':512,
           'w':0.5,
           'use_attention' : True
        }
    },
    'stage2': {
        'Name': 'stage2',
        'Data': './data/summery_new.txt',
        'Mode': 1,
        'Lr': 0.001,
        'Epoch': 300,
        'Dataset_mode': 'uid',
        'Model': GNN,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': train_gcn,
    },
    'stage2_1': {
        'Name': 'stage2_1',
        'Data': './data/summery_new.txt',
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'uid',
        'Model': GNN,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': train_gcn,
        
    },
    'stage2_2': {
        'Name': 'stage2_2',
        'Data': './data/summery_new.txt',
        'Lr': 0.001,
        'Epoch': 300,
        'Dataset_mode': 'uid',
        'Model': GNN,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': train_gcn,
        
    },
    'stage2_3': {
        'Name': 'stage2_3',
        'Data': './data/summery_new.txt',
        'Mode': 3,
        'Lr': 0.001,
        'Epoch': 300,
        'Dataset_mode': 'uid',
        'Model': GNN,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': train_gcn,
        
    },
    'stage2_4': {
        'Name': 'stage2_4',
        'Data': './data/summery_new.txt',
        'Mode': 4,
        'Lr': 0.001,
        'Epoch': 300,
        'Dataset_mode': 'uid',
        'Model': GNN,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': train_gcn,
        
    },
    'Multi_IBNet': {
        'Name': 'Multi_IBNet',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'radio_img_label',
        'Model': Multi_IBNet,
        'Optimizer': Adam,
        'Loss': joint_loss,
        'Run': run,
        'w1': 0.1,
        'w2': 1,
        'w3': 0,
        'Input_args': {
           'vision_encoder_selection': 2,
           'vision_dim':2048,
           'w':0.5
        }
    },
    'VAE_CLNet': {
        'Name': 'VAE_CLNet',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'radio_img_label',
        'Model': VAE_CLNet,
        'Optimizer': Adam,
        'Loss': joint_loss_2,
        'Run': run_0,
        'Input_args': {
           'vision_encoder_selection': 2,
           'vision_dim':2048,
           'w':0.5
        }
    },
    'img_2D': {
        'Name': 'img_2D',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'img',
        'Model': Resnet_2D,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_img
    },
    'c_mib': {
        'Name': 'c_mib',
        'Data': './data/summery_new.txt',
        'Batch': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'Dataset_mode': 'new_data',
        'Model': MIB,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_cmib,
    },
    'mpgsurv': {
        'Name': 'mpgsurv',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'all_model',
        'Model': ResNet_Linformer,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_mpgsurv,
    },
}
