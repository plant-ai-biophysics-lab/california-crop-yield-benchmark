import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import numpy as np
import math
import sys
import time
from pathlib import Path
import torch
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union

from src.SimCLRDataLoader import get_simclr_data_loader
from loss import losses
from src import resnet_simclr 
from src.YieldB import YieldBenchmark

device = "cuda" if torch.cuda.is_available() else "cpu"
from src.configs import set_seed 
set_seed(0)


class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):

        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True

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
                    criterion = losses.ContrastiveLoss(zi.shape[0], device)
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

class FineTune():

    def __init__(self, 
                 exp_name: str, 
                 device: str,
                 ):
        self.exp_name = exp_name
        self.device = device

        self.exp_output_dir = '/data2/hkaman/Projects/Foundational/EXPs/' + 'EXP_' + exp_name
        os.makedirs(self.exp_output_dir, exist_ok=True)
        self.best_model_name = os.path.join(self.exp_output_dir, 'best_model_' + exp_name + '.pth')
        # self.best_checkpoint_dir = os.path.join(self.exp_output_dir, 'best_checkpoints_' + self.exp + '.pth')
        # self.checkpoint_dir = os.path.join(self.exp_output_dir, 'checkpoints')
        self.loss_dir = os.path.join(self.exp_output_dir, 'loss')
        os.makedirs(self.loss_dir, exist_ok=True)

        self.loss_fig_name = os.path.join(self.loss_dir, 'loss_' + exp_name + '.png')
        self.loss_df_name = os.path.join(self.loss_dir, 'loss_' + exp_name + '.csv')

        self.train_df_name = os.path.join(self.exp_output_dir, exp_name + '_train.csv')
        self.valid_df_name = os.path.join(self.exp_output_dir, exp_name + '_valid.csv')
        self.test_df_name = os.path.join(self.exp_output_dir, exp_name + '_test.csv')

    def fit(self, config: Union[Dict]):
        
        model = YieldBenchmark(config) 
        model.to(self.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameteres: {num_params}  ")
        print(f"****************************************************************")


        return model
    
    def train(self, 
              model: torch.nn.Module,  
              dataloader_train: Iterable,   
              dataloader_valid: Iterable,             
              optimizer: str, 
              loss: str, 
              lr: float, 
              wd: float,
              epochs: int, 
            ):
        

        param_groups = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            param_groups, 
            lr =  lr, 
            weight_decay = wd,
            betas=(0.9, 0.95)
            )

        best_val_loss  = 1e15 # initial dummy value
        early_stopping = EarlyStopping(tolerance = 50, min_delta = 50)
        loss_stats = {'train': [],"val": []}
        

        for epoch in range(1, epochs + 1):

            training_start_time = time.time()
            train_epoch_loss = 0
            model.train()

            for batch, sample in enumerate(dataloader_train):
                
                list_ytrain_pred = model(                    
                    landsat = sample['landsat_linear'].to(device), 
                    et = sample['et_linear'].to(device), 
                    climate = sample['climate_linear'].to(device), 
                    soil = sample['soil_linear'].to(device)) 
                
                optimizer.zero_grad()
                ytrain_true = sample['yield'].to(device)
                train_loss = self._calculate_timeseries_loss(ytrain_true, list_ytrain_pred, loss)

                train_loss.backward()

                optimizer.step()
                train_epoch_loss += train_loss.item() 

            # VALIDATION    
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_epoch_loss = 0
                for batch, sample in enumerate(dataloader_valid):
                
                    list_yvalid_pred = model(                    
                        landsat = sample['landsat_linear'].to(device), 
                        et = sample['et_linear'].to(device), 
                        climate = sample['climate_linear'].to(device), 
                        soil = sample['soil_linear'].to(device))   
                    
                    yvalid_true = sample['yield'].to(device)
                    valid_loss = self._calculate_timeseries_loss(yvalid_true, list_yvalid_pred, loss)

                    val_epoch_loss += valid_loss.item()

            loss_stats['train'].append(train_epoch_loss/len(dataloader_train))
            loss_stats['val'].append(val_epoch_loss/len(dataloader_valid))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03} [{training_duration_time:.3f} (s)]: Train MSE Loss: {train_epoch_loss/len(dataloader_train):.4f} | Val MSE Loss: {val_epoch_loss/len(dataloader_valid):.4f}') 

            if (val_epoch_loss/len(dataloader_valid)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(dataloader_valid))
                torch.save(model.state_dict(), self.best_model_name)
                print(f'=============================== Best model Saved! Val MSE: {best_val_loss:.4f}')
                status = True
            else:
                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                break

        self._save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)

    def predict(self, model, data_loader, category: str):

        model.load_state_dict(torch.load(self.best_model_name))

        output_files =[]
        with torch.no_grad():
            for batch, sample in enumerate(data_loader):

                ypred_list = model(
                        landsat = sample['landsat_linear'].to(device), 
                        et = sample['et_linear'].to(device), 
                        climate = sample['climate_linear'].to(device), 
                        soil = sample['soil_linear'].to(device)
                ) 

                years = sample['year']
                county_names = sample['county']
                crop_names = sample['crop_name']
                ytrue_list = sample['yield']


                this_batch = {"year": years,
                              "county_names": county_names, 
                              "crop_names": crop_names, 
                              "ytrue": ytrue_list.detach().cpu().numpy()}

                for i, pred in enumerate(ypred_list):
                    if len(ypred_list) == 1:
                        key = "ypred_m12"  
                    else:
                        key = f"ypred_m{i+1}"  


                    this_batch[key] = pred.detach().cpu().numpy()

                output_files.append(this_batch)

            modified_df = self._return_modified_pred_df(output_files, None, 16)


            if category == 'train':
                name_tr = self.train_df_name[:-4]  + '.csv'
                modified_df.to_csv(name_tr)
                print("train inference is done!")

            elif category == 'valid':
                name_val = self.valid_df_name[:-4]  + '.csv'
                modified_df.to_csv(name_val)
                print("validation inference is done!")
                
            elif category == 'test':
                name_te = self.test_df_name[:-4] + '.csv'
                modified_df.to_csv(name_te)
                print("test inference is done!")
    
    def _return_modified_pred_df(self, pred_npy, blocks_list, wsize=None):


        columns = ['year', 'county', 'crop_name', 'ytrue']
        data = {col: [] for col in columns}  # Initialize dictionary for DataFrame
        pred_keys = [key for key in pred_npy[0].keys() if key.startswith('ypred')]
        for key in pred_keys:
            data[key] = []


        for l in range(len(pred_npy)):
            year = pred_npy[l]['year']
            data['year'].append(year)
            county = pred_npy[l]['county_names']
            data['county'].append(county)
            crop_name = pred_npy[l]['crop_names']
            data['crop_name'].append(crop_name)
            ytrue = pred_npy[l]['ytrue']
            data['ytrue'].append(ytrue)

            for key in pred_keys:
                flattened_pred = pred_npy[l][key].flatten()
                data[key].append(flattened_pred)

        empty_dict = {key: None for key in data.keys()}

        for key in data:
            if data[key]:  
                output = np.concatenate(data[key])
                
                empty_dict[key] = output

        results = pd.DataFrame(empty_dict)
        return results
    
    def _save_loss_df(self, loss_stat, loss_df_name, loss_fig_name):

        df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        df.to_csv(loss_df_name) 
        plt.figure(figsize=(12,8))
        sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
        plt.ylim(0, df['value'].max())
        plt.savefig(loss_fig_name, dpi = 300)

    def _save_checkpoint(self, state, filename="checkpoint.pth"):
        """
        Saves a model checkpoint during training.

        Parameters:
        - state (dict): State to save, including model and optimizer states.
        - filename (str): File name to save the checkpoint.
        """
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, model, optimizer):
        """
        Loads a checkpoint into a model and optimizer.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        - model (nn.Module): Model to load the checkpoint into.
        - optimizer (torch.optim): Optimizer to load the checkpoint into.
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_val_loss']

    def _calculate_loss(self, y_pred, y_true, loss_type):

        if loss_type == 'mse':
            return losses.mse_loss(y_pred, y_true)
        elif loss_type == 'wmse':
            return losses.weighted_mse_loss(y_pred, y_true)

    def _calculate_timeseries_loss(self, y_true, list_y_pred, loss_type):
        """
        Calculates the cumulative mean squared error loss for a list of predictions or a single prediction.

        Parameters:
        - y_true (Tensor): The ground truth values.
        - list_y_pred (list of Tensors or Tensor): List of predicted values or a single tensor.
        - loss_type (str): The type of loss ('mse', 'wmse', 'huber').

        Returns:
        - Tensor: The cumulative MSE loss for all predictions.
        """

        if isinstance(list_y_pred, list):
            losses_list = [self._calculate_loss(y_pred, y_true, loss_type) for y_pred in list_y_pred]
            total_loss = sum(losses_list)
        else:
            # list_y_pred is a single tensor
            y_pred = list_y_pred
            total_loss = self._calculate_loss(y_pred, y_true, loss_type)

        return total_loss

class ViTYieldEst:
    """
    This class implements the training and evaluation of a neural network model for yield estimation.

    Attributes:
        model (nn.Module): The neural network model.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for regularization.
        exp (str): Experiment identifier used for file naming and directory structuring.
        optimizer (torch.optim): Optimizer for training the model.
        exp_output_dir (str): Directory for experiment outputs.
        best_model_name (str): Path for saving the best model.
        checkpoint_dir (str): Directory for saving checkpoints.
        loss_fig_name (str): File name for the loss figure.
        loss_df_name (str): File name for the loss data in CSV format.
        train_df_name (str): File name for the training data.
        valid_df_name (str): File name for the validation data.
        test_df_name (str): File name for the test data.
        timeseries_fig (str): File name for the time series figure.
        scatterplot (str): File name for the scatter plot.

    Methods:
        train: Trains the model using the provided data loaders.
        predict: Generates predictions using the trained model.
        _return_pred_df: Helper function to process prediction results into a DataFrame.
        xy_vector_generator: Generates X and Y coordinate vectors for a given point and window size.
    """

    def __init__(self, model, lr: float, wd: float, exp: str):
        # Initialize model parameters and directories
        self.model = model
        self.lr = lr
        self.wd = wd
        self.exp = exp

        params = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = torch.optim.AdamW(
                params,               # Model parameters to optimize
                lr=self.lr,              # Lower learning rate, you can experiment with values like 1e-4, 1e-5, etc.
                # betas=(0.9, 0.98),    # Adjust beta values, slower decay of the running averages
                weight_decay=self.wd,    # L2 regularization strength
                # amsgrad=True          # Use the AMSGrad variant of AdamW
                )

        self.exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + self.exp

        self.best_model_name = os.path.join(self.exp_output_dir, 'best_model_' + self.exp + '.pth')
        self.last_model_name = os.path.join(self.exp_output_dir, 'last_model_' + self.exp + '.pth')
        self.best_checkpoint_dir = os.path.join(self.exp_output_dir, 'best_checkpoints_' + self.exp + '.pth')
        self.checkpoint_dir = os.path.join(self.exp_output_dir, 'checkpoints')
        self.loss_fig_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.png')
        self.loss_df_name = os.path.join(self.exp_output_dir, 'loss', 'loss_' + self.exp + '.csv')
        self.train_df_name = os.path.join(self.exp_output_dir, self.exp + '_train.csv')
        self.valid_df_name = os.path.join(self.exp_output_dir, self.exp + '_valid.csv')
        self.test_df_name = os.path.join(self.exp_output_dir, self.exp + '_test.csv')
        self.timeseries_fig = os.path.join(self.exp_output_dir, self.exp + '_timeseries.png')
        self.scatterplot = os.path.join(self.exp_output_dir, self.exp + '_scatterplot.png')

    def train(self, data_loader_training, data_loader_validate, 
              loss: str, 
              epochs: int, 
              loss_stop_tolerance: int):
        """
        Trains the model using the provided training and validation data loaders.

        Parameters:
        - data_loader_training: DataLoader for the training dataset.
        - data_loader_validate: DataLoader for the validation dataset.
        - loss_type (str): Type of loss function to use ('mse', 'wmse', 'huber', 'wass').
        - epochs (int): Number of epochs to train the model.
        - loss_stop_tolerance (int): Early stopping tolerance level.
        """

        best_val_loss  = 1e15 # initial dummy value
        early_stopping = EarlyStopping(tolerance = loss_stop_tolerance, min_delta = 50)
        loss_stats = {'train': [],"val": []}
        

        for epoch in range(1, epochs + 1):

            training_start_time = time.time()
            train_epoch_loss = 0
            self.model.train()

            for batch, sample in enumerate(data_loader_training):
                                
                list_ytrain_pred, _ = self.model(img = sample['image'].to(device), 
                                                 context = sample['EmbText'], 
                                                 met = sample['met'].to(device), 
                                                 yz = sample['YZ'].to(device)) 

                self.optimizer.zero_grad()
                ytrain_true = sample['mask'][:,:,:,:,0].to(device)
                train_loss = self._calculate_timeseries_loss(ytrain_true, list_ytrain_pred, loss, sample['weight'].to(device))

                train_loss.backward()

                self.optimizer.step()
                train_epoch_loss += train_loss.item() 

            # VALIDATION    
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_epoch_loss = 0
                for batch, sample in enumerate(data_loader_validate):
                
                    list_yvalid_pred, _ = self.model(img = sample['image'].to(device), 
                                                     context = sample['EmbText'], 
                                                     met = sample['met'].to(device), 
                                                     yz = sample['YZ'].to(device),)   
                    
                    yvalid_true = sample['mask'][:,:,:,:,0].to(device)
                    valid_loss = self._calculate_timeseries_loss(yvalid_true, list_yvalid_pred, loss, sample['weight'].to(device))

                    val_epoch_loss += valid_loss.item()

            loss_stats['train'].append(train_epoch_loss/len(data_loader_training))
            loss_stats['val'].append(val_epoch_loss/len(data_loader_validate))

            training_duration_time = (time.time() - training_start_time)        
            print(f'Epoch {epoch+0:03} [{training_duration_time:.3f} (s)]: Train MSE Loss: {train_epoch_loss/len(data_loader_training):.4f} | Val MSE Loss: {val_epoch_loss/len(data_loader_validate):.4f}') 

            # checkpoint = {
            # 'epoch': epoch + 1,
            # 'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'best_val_loss': best_val_loss
            # }

            # self._save_checkpoint(checkpoint, filename= os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            if (val_epoch_loss/len(data_loader_validate)) < best_val_loss or epoch==0:
                        
                best_val_loss=(val_epoch_loss/len(data_loader_validate))
                torch.save(self.model.state_dict(), self.best_model_name)

                # self._save_checkpoint(checkpoint, filename = self.best_checkpoint_dir)

                # early_stopping.update(False)
                print(f'=============================== Best model Saved! Val MSE: {best_val_loss:.4f}')

                status = True


            else:

                status = False

            early_stopping(status)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch:", epoch)
                torch.save(self.model.state_dict(), self.last_model_name)
                break

        self._save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)

    def predict(self, model, data_loader, category: str):

        # model = ClimMgmtAware_ViT(config).to(device)
        model.load_state_dict(torch.load(self.best_model_name))

        output_files =[]
        with torch.no_grad():
            for batch, sample in enumerate(data_loader):

                pred_list, text_attn_list = self.model(img = sample['image'].to(device), 
                                        context =  sample['EmbText'], 
                                        met = sample['met'].to(device), 
                                        yz = sample['YZ'].to(device)) 

                # if category == 'train':
                #     np.save(os.path.join(self.exp_output_dir, f'attn_scores/train_attn_scores_{batch}.npy'), text_attn_list[0].detach().cpu().numpy())

                #     # np.save(f'/data2/hkaman/Projects/ViT/EXPs/July/attnscores/train_attn_scores_{batch}.npy', text_attn_list[0].detach().cpu().numpy())

                block_id = sample['block']
                block_cultivar_id = sample['cultivar']
                block_x_coords = sample['X']
                block_y_coords = sample['Y']
                this_batch = {"block": block_id, 
                                    "cultivar": block_cultivar_id, 
                                    "X": block_x_coords, "Y": block_y_coords,
                                    "ytrue": sample['mask'].detach().cpu().numpy()}

                for i, pred in enumerate(pred_list):
                    if len(pred_list) == 1:
                        key = "ypred_w15"  
                    else:
                        key = f"ypred_w{i+1}"  # Creates keys like ypred_w1, ypred_w2, ..., ypred_wN
                    this_batch[key] = pred.detach().cpu().numpy()

                output_files.append(this_batch)

            modified_df = self._return_modified_pred_df(output_files, None, 16)
            if category == 'train':
                name_tr = self.train_df_name[:-4]  + '.csv'
                modified_df.to_csv(name_tr)
                print("train inference is done!")

            elif category == 'valid':
                name_val = self.valid_df_name[:-4]  + '.csv'
                modified_df.to_csv(name_val)
                print("validation inference is done!")
                
            elif category == 'test':
                name_te = self.test_df_name[:-4] + '.csv'
                modified_df.to_csv(name_te)
                print("test inference is done!")

    def _return_modified_pred_df(self, pred_npy, blocks_list, wsize=None):
        if blocks_list is None: 
            all_block_names = [dict['block'] for dict in pred_npy]#[0]
            blocks_list = list(set(item for sublist in all_block_names for item in sublist))


        OutDF = pd.DataFrame()
        columns = ['block', 'cultivar', 'x', 'y', 'ytrue']
        data = {col: [] for col in columns}  # Initialize dictionary for DataFrame

        # Initialize lists for predictions dynamically based on the first item's keys
        pred_keys = [key for key in pred_npy[0].keys() if key.startswith('ypred')]
        for key in pred_keys:
            data[key] = []

        for block in blocks_list:
            name_split = os.path.split(block)[-1]
            block_name = name_split.replace(name_split[7:], '')
            root_name = name_split.replace(name_split[:4], '').replace(name_split[3], '')
            block_id = root_name
            
            res = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {block_name}}
            list_d = res.get(block_name)
            cultivar_id = list_d[1]
        
            for l in range(len(pred_npy)):
                tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
                if len(tb_pred_indices) !=0:   
                    for index in tb_pred_indices:

                        x0 = pred_npy[l]['X'][index]
                        y0 = pred_npy[l]['Y'][index]
                        x_vector, y_vector = self._xy_vector_generator(x0, y0, wsize)
                        data['x'].append(x_vector)
                        data['y'].append(y_vector)
                        data['ytrue'].append(pred_npy[l]['ytrue'][index].flatten())

                        tb_block_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[block_id], dtype=np.int32)
                        data['block'].append(tb_block_id)

                        tb_cultivar_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[cultivar_id], dtype=np.int8)
                        data['cultivar'].append(tb_cultivar_id)



                        # Handle predictions dynamically
                        for key in pred_keys:
                            flattened_pred = pred_npy[l][key][index].flatten()
                            data[key].append(flattened_pred)

        empty_dict = {key: None for key in data.keys()}
        # Convert lists to numpy arrays for consistency
        for key in data:
            if data[key]:  # Ensure there's data to concatenate
                # print(len(data[key]))
                output = np.concatenate(data[key])
                empty_dict[key] = output
                # print(key, output.shape)

        # Create DataFrame from data dictionary
        OutDF = pd.DataFrame(empty_dict)
        return OutDF
    
    def _xy_vector_generator(self, x0, y0, wsize):

        x_vector, y_vector = [], []
        
        for i in range(x0, x0+wsize):
            for j in range(y0, y0+wsize):
                x_vector.append(i)
                y_vector.append(j)

        return x_vector, y_vector 
    
    def _save_loss_df(self, loss_stat, loss_df_name, loss_fig_name):

        df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        df.to_csv(loss_df_name) 
        plt.figure(figsize=(12,8))
        sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
        plt.ylim(0, df['value'].max())
        plt.savefig(loss_fig_name, dpi = 300)

    def _save_checkpoint(self, state, filename="checkpoint.pth"):
        """
        Saves a model checkpoint during training.

        Parameters:
        - state (dict): State to save, including model and optimizer states.
        - filename (str): File name to save the checkpoint.
        """
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, model, optimizer):
        """
        Loads a checkpoint into a model and optimizer.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        - model (nn.Module): Model to load the checkpoint into.
        - optimizer (torch.optim): Optimizer to load the checkpoint into.
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_val_loss']

    def _calculate_loss(self, y_pred, y_true, loss_type, weight):

        if loss_type == 'mse':
            return losses.mse_loss(y_pred, y_true)
        elif loss_type == 'wmse':
            return losses.weighted_mse_loss(y_pred, y_true, weight)

    def _calculate_timeseries_loss(self, y_true, list_y_pred, loss_type, weights):
        """
        Calculates the cumulative mean squared error loss for a list of predictions or a single prediction.

        Parameters:
        - y_true (Tensor): The ground truth values.
        - list_y_pred (list of Tensors or Tensor): List of predicted values or a single tensor.
        - loss_type (str): The type of loss ('mse', 'wmse', 'huber').

        Returns:
        - Tensor: The cumulative MSE loss for all predictions.
        """

        if isinstance(list_y_pred, list):
            losses_list = [self._calculate_loss(y_pred, y_true, loss_type, weights) for y_pred in list_y_pred]
            total_loss = sum(losses_list)
        else:
            # list_y_pred is a single tensor
            y_pred = list_y_pred
            total_loss = self._calculate_loss(y_pred, y_true, loss_type, weights)

        return total_loss

                

