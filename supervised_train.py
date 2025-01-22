import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import datetime
import os
import sys
import time
from pathlib import Path
import torch
from typing import Iterable
from src.configs import set_seed 
set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

from dataset import datalaoder
from utils import misc
from src import resnet_simclr 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch.nn as nn


# Define a model wrapper with an extra classification layer
class FineTunedResNetSimCLR(nn.Module):
    def __init__(self, base_model, num_classes=8):
        super(FineTunedResNetSimCLR, self).__init__()
        self.base_model = base_model  # Load the pretrained model
        # Add an additional fully connected layer to go from 128 to 8 classes
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Pass through the base model (up to 128-dimensional output)
        features = self.base_model(x)
        # Pass through the classifier layer for 8-class output
        out = self.classifier(features)
        return out
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

def main(args):

    data_loader_training, data_loader_validate, data_loader_test = datalaoder.get_dataloaders(
            batch_size = 128, 
            exp_name = args.exp_name)
    
    base_model = resnet_simclr.ResNetSimCLR(base_model='resnet50', out_dim=args.out_dim).to(device)
    model = FineTunedResNetSimCLR(base_model=base_model, num_classes=8).to(device)

    model.to(device)
    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr =  args.lr, 
        betas=(0.9, 0.95)
        )
    criterion = torch.nn.CrossEntropyLoss().to(device)


    loss_stats = {'train': [],"val": []}
    best_val_loss  = 1e15


    for epoch in range(0, args.epochs):
        training_start_time = time.time()
        train_epoch_loss = 0
        top1_train_accuracy, top5_train_accuracy = 0, 0


        model.train()

        for batch, (x_batch, y_batch) in enumerate(data_loader_training):
                                
            optimizer.zero_grad()

            x_batch = x_batch[..., -1].to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            train_loss = criterion(logits, y_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_train_accuracy += top1[0]
            top5_train_accuracy += top5[0]

            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item() 

        # VALIDATION    
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_epoch_loss = 0
            top1_valid_accuracy, top5_valid_accuracy = 0, 0
            for batch_val, (x_batch_val, y_batch_val) in enumerate(data_loader_validate):
                
                valid_logits = model(x_batch_val[..., -1].to(device))
                valid_loss = criterion(valid_logits, y_batch_val.to(device))
                val_epoch_loss += valid_loss.item()

                top1_valid, top5_valid = accuracy(logits, y_batch, topk=(1, 5))
                top1_valid_accuracy += top1_valid[0]
                top5_valid_accuracy += top5_valid[0]

        top1_train_accuracy /= (batch + 1)
        top5_train_accuracy /= (batch + 1)
        top1_valid_accuracy /= (batch_val + 1)
        top5_valid_accuracy /= (batch_val + 1)

        loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
        loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

        training_duration_time = (time.time() - training_start_time)        
        print(f'Epoch {epoch+0:03} [{training_duration_time:.3f} (s)]: Train CE Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val CE Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 
        print(f'\t Train Top1 acc: {top1_train_accuracy.item():.2f}\tTop5 acc: {top5_train_accuracy.item():.2f} | \t Valid Top1 acc {top1_valid_accuracy.item():.2f}\tTop5 acc: {top5_valid_accuracy.item():.2f}') 
        print(f'===========================================================================================') 

        if args.exp_name and epoch + 1 == args.epochs:
                    
            root_exp_dir = '/data2/hkaman/Projects/Foundational/' 
            exp_output_dir = root_exp_dir + 'EXPs/' + 'EXP_' + args.exp_name
            best_model_name = os.path.join(exp_output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_name)


    loss_fig_name = os.path.join(exp_output_dir, 'loss.png')
    loss_df_name = os.path.join(exp_output_dir, 'loss.csv')
    
    df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 

    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name", type=str, default = "test", help = "Experiment name")
    parser.add_argument("--batch_size", type=int, default = 512, help = "Batch size")
    parser.add_argument("--lr", type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd", type=float, default = 0.001, help = "Value of weight decay")
    parser.add_argument("--epochs", type=int, default = 300, help = "The number of epochs")
    parser.add_argument("--loss", type=str, default = "mse", help = "Loss function")
    parser.add_argument("--out_dim", type=int, default = 128, help = "The output feature dimension")

    args = parser.parse_args()

    main(args)


    


