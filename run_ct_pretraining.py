import torch,argparse,os
from utils.utils import seed_everything
from utils.Constractive import ContrastiveLoss
from Net.networks import *
from data.dataset import Liver_dataset, Liver_normalization_dataset
from torch.utils.data import random_split
from Net.api import *
from utils.observer import Runtime_Observer
from pathlib import Path

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    dataset = Liver_normalization_dataset("./data/summery_new.txt", mode='SimCLR')
    train_ratio = 0.75
    train_dataset, val_dataset = random_split(dataset, [int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))]) 

    trainDataLoader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False) 

    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''
    Training logs and monitors
    '''
    target_dir = Path('./logs/')
    target_dir.mkdir(exist_ok=True)
    target_dir = target_dir.joinpath('SimCLR')
    target_dir.mkdir(exist_ok=True)
    target_dir = target_dir.joinpath('Resnet50')
    target_dir.mkdir(exist_ok=True)
    checkpoints_dir = target_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = target_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    observer = Runtime_Observer(log_dir=log_dir, checkpoints_dir= checkpoints_dir, device=device, name='SimCLR', seed= args.seed)

    observer.log(f'[DEBUG]Observer init successfully, program start\n')


    model = CTEncoder().to(device)

    contrastive_loss = ContrastiveLoss(temperature=args.temperature, k=args.k).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    training_loop_contrastive(observer, args.max_epoch, trainDataLoader, valDataLoader, model, device, optimizer, criterion = contrastive_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--max_epoch', default=30, type=int, help='')
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default='model/vision_encoder_resnet50.pth')
    args = parser.parse_args()

    train(args)
