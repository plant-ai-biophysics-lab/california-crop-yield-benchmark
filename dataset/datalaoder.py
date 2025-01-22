import os
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision.transforms import transforms
from torchvision import transforms
from dataset.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
from einops import rearrange


from src.configs import set_seed
set_seed(0)


EXTREME_LOWER_THRESHOLD = 9  #22.24
EXTREME_UPPER_THRESHOLD = 22 #54.36
HECTARE_TO_ACRE_SCALE = 2.471 # 2.2417





def get_dataloaders(
        batch_size:int, 
        exp_name: str): 

    
    root_exp_dir = '/data2/hkaman/Projects/Foundational/' 
    exp_output_dir = root_exp_dir + 'EXPs/' + 'EXP_' + exp_name

    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(os.path.join(exp_output_dir, 'loss'))

    train_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/train.csv', index_col=0)
    valid_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/val.csv', index_col= 0)
    test_csv  = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/test.csv', index_col= 0)

    print(f"{train_csv.shape} | {valid_csv.shape} | {test_csv.shape}")
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#
    #csv_coord_dir = '/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/'


    dataset_training = Sentinel_Dataset(
        train_csv
    )

    dataset_validate = Sentinel_Dataset(
        valid_csv   
    )
    
    dataset_test = Sentinel_Dataset(
        test_csv
    )     

    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
  
    data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                    shuffle=False, num_workers=8)  
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False, num_workers=8) 

    return data_loader_training, data_loader_validate, data_loader_test



class Sentinel_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df.reset_index(inplace=True, drop=True)
        self.wsize = 16
        
        # Create a mapping for unique labels
        unique_labels = sorted(self.df['cultivar_id'].unique())
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.df)
    
    def _crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        if src.ndim == 2:
            src = np.expand_dims(src, axis=0)
            src = np.expand_dims(src, axis=-1)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
    
    def histogram_equalization_4d(self, image):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        eq_image = np.empty_like(image)
        for c in range(image.shape[0]):
            for t in range(image.shape[3]):
                eq_image[c, :, :, t] = cv2.equalizeHist(image[c, :, :, t])
        return eq_image

    def __getitem__(self, idx):
        xcoord = self.df.loc[idx]['X'] 
        ycoord = self.df.loc[idx]['Y'] 
        S2_path = self.df.loc[idx]['IMG_PATH']
        x = self._crop_gen(S2_path, xcoord, ycoord) 
        x = np.swapaxes(x, -1, 0)   
        x = self.histogram_equalization_4d(x)
        x = x / 255.0
        x = torch.as_tensor(x, dtype=torch.float32)

        # Original label
        original_y = self.df.loc[idx]['cultivar_id']
        # Remap label to a consecutive range
        y = self.label_mapping[original_y]
        y = torch.tensor(y, dtype=torch.long)

        return x, y
