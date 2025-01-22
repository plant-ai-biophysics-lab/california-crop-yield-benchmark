import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import datetime
import math
import sys
import time
from pathlib import Path
import torch
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


device = "cuda" if torch.cuda.is_available() else "cpu"
from dataset.SimCLRDataLoader import get_simclr_data_loader
from loss.losses import ContrastiveLoss
from src.configs import set_seed 
from src import resnet_simclr 
set_seed(0)



class Pretrain():
    def __init__(self, exp_name: str, device
                 ):
        self.exp_name = exp_name
        self.device = device

    def fit(self, 
            base_model: str, 
            out_dim: int, 
            ):
        

        if base_model == 'resnet50': 

            model = resnet_simclr.ResNetSimCLR(
                base_model = 'resnet50', 
                out_dim = out_dim) 
            model.to(self.device)
        
        
        return model
    

    def train(self, 
              model: torch.nn.Module,  
              dataloader: Iterable,           
              optimizer: str, 
              loss: str, 
              lr: float, 
              wd: float,
              epochs: int, 
              batch_size: int,
            ):
        

        param_groups = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr =  lr, 
            betas=(0.9, 0.95)
            )

        start_time = time.time()
        loss_stats = {'train': []}
        for epoch in range(0, epochs):
            training_start_time = time.time()
            train_stats = self.train_one_epoch(
                model = model, 
                dataloader = dataloader, 
                optimizer = optimizer, 
                loss = loss,
                device = self.device, 
                batch_size = batch_size
            )

            training_duration_time = (time.time() - training_start_time)
            print(f"Epoch {epoch}[{training_duration_time:.3f} (s)]: Training Contrastive Loss = {train_stats:.3f}")
            if self.exp_name and epoch + 1 == epochs:
                root_exp_dir = '/data2/hkaman/Projects/Foundational/' 
                exp_output_dir = root_exp_dir + 'EXPs/' + 'EXP_' + self.exp_name
                best_model_name = os.path.join(exp_output_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_name)


            loss_stats['train'].append(train_stats)


        loss_fig_name = os.path.join(exp_output_dir, 'loss.png')
        loss_df_name = os.path.join(exp_output_dir, 'loss.csv')
        
        df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        df.to_csv(loss_df_name) 

        plt.figure(figsize=(12,8))
        sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
        plt.ylim(0, df['value'].max())
        plt.savefig(loss_fig_name, dpi = 300)


    def train_one_epoch(self, 
                        model: torch.nn.Module,
                        dataloader: Iterable, 
                        optimizer: torch.optim.Optimizer, 
                        loss: str,
                        device: torch.device,
                        batch_size: int):
    
        model.train(True)
        batch_size = batch_size
        optimizer.zero_grad()
        train_epoch_loss = 0
        total_step = len(dataloader) - 1

        iter = 0
        for data_iter_step, x in enumerate(dataloader):

            # prevent the number of grids from being too large to cause out of memory
            train_loader = get_simclr_data_loader(x, batch_size=batch_size)

            for xi, xj in train_loader:
                xi = xi.to(device, non_blocking=True)
                xj = xj.to(device, non_blocking=True)

                zi = model(xi)
                zj = model(xj)

                if loss == 'contrastive':
                    criterion = ContrastiveLoss(zi.shape[0], device)
                loss = criterion(zi, zj)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)


                train_epoch_loss += loss_value

                iter += 1


        final_loss_value  = train_epoch_loss / iter

        return final_loss_value





                

