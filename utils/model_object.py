from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from Net.loss_functions import *
from Net.api import *
from Net.networks import *
from Net.cp_networks import *

models = {
    'MDIB': {
        'Name': 'MDIB',
        'Data': './data/summery_new.txt',
        'Batch': 2,
        'Lr': 0.0001,
        'Epoch': 300,
        'Dataset_mode': 'new',
        'Model': MDIB,
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
}
