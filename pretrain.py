import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from src.configs import set_seed 
set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

from dataset.SimCLRDataLoader import pretrain_dataloader
from utils import misc
import argparse


def main(args):

    data_loader_sentinel = pretrain_dataloader(
        exp_name = args.exp_name, 
        batch_size = args.batch_size)
    
    PreTrain = misc.Pretrain(exp_name = args.exp_name, device = device)
    model = PreTrain.fit(base_model= 'resnet50', out_dim= 128)

    PreTrain.train(model = model, 
                   dataloader = data_loader_sentinel, 
                   optimizer= args.optimizer, 
                   loss = args.loss, 
                   lr = args.lr,
                   wd = args.wd, 
                   epochs= args.epochs,
                   batch_size= args.batch_size)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name", type=str, default = "test", help = "Experiment name")
    parser.add_argument("--batch_size", type=int, default = 512, help = "Batch size")
    parser.add_argument("--lr", type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd", type=float, default = 0.0001, help = "Value of weight decay")
    parser.add_argument("--epochs", type=int, default = 300, help = "The number of epochs")
    parser.add_argument("--loss", type=str, default = "contrastive", help = "Loss function")
    parser.add_argument("--optimizer", type=str, default = "adamw", help = "Optimizer")
    parser.add_argument("--out_dim", type=int, default = 128, help = "The output feature dimension")

    args = parser.parse_args()

    main(args)