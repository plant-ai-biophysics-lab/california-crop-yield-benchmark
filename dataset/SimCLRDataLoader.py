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


def pretrain_dataloader(
        exp_name :str, 
        batch_size: int):
    dataset_sentinel = Sentinel_Dataset(exp_name= exp_name)
    sampler_sentinel = torch.utils.data.RandomSampler(dataset_sentinel)

    data_loader_sentinel = torch.utils.data.DataLoader(
        dataset_sentinel, 
        sampler = sampler_sentinel,
        batch_size = batch_size,
        num_workers = 8,
        shuffle = False,
        drop_last = True,
    )

    return data_loader_sentinel

class Sentinel_Dataset(Dataset):

    def __init__(self, exp_name: str):


        root_exp_dir = '/data2/hkaman/Projects/Foundational/' 
        exp_output_dir = root_exp_dir + 'EXPs/' + 'EXP_' + exp_name

        isExist  = os.path.isdir(exp_output_dir)

        if not isExist:
            os.makedirs(exp_output_dir)
            os.makedirs(os.path.join(exp_output_dir, 'loss'))


        # self.df = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/train.csv', index_col=0)
        train_df = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/train.csv', index_col=0)
        valid_df = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/val.csv', index_col=0)
        test_df = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/test.csv', index_col=0)

        # Concatenate dataframes
        self.df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
        self.df.reset_index(inplace = True, drop = True)
        self.wsize = 16

    def __len__(self):
        return len(self.df)
    

    def _crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        if src.ndim == 2:
            src = np.expand_dims(src, axis = 0)
            src = np.expand_dims(src, axis = -1)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
    
    def histogram_equalization_4d(self, image):
        """
        Apply histogram equalization to each channel of each timeseries frame in the image,
        ensuring each slice is an 8-bit single-channel image.

        Args:
        - image (numpy.ndarray): Input image array with values normalized to 0-255 and shape (C, H, W, T).

        Returns:
        - (numpy.ndarray): The histogram equalized image.
        """
        # Check if image dtype is uint8, convert if necessary
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Prepare the output array with the same shape
        eq_image = np.empty_like(image)

        # Iterate over each channel and timeseries
        for c in range(image.shape[0]):  # For each channel
            for t in range(image.shape[3]):  # For each time point
                # Apply histogram equalization to each slice (channel, :, :, timeseries)
                eq_image[c, :, :, t] = cv2.equalizeHist(image[c, :, :, t])

        return eq_image

    def __getitem__(self, idx):

        xcoord = self.df.loc[idx]['X'] 
        ycoord = self.df.loc[idx]['Y'] 
        S2_path = self.df.loc[idx]['IMG_PATH']
        x = self._crop_gen(S2_path, xcoord, ycoord) 
        x = np.swapaxes(x, -1, 0)   
        x = self.histogram_equalization_4d(x)
        x = x / 255.
        x = torch.as_tensor(x, dtype=torch.float32)

        return x
    
class Sentinel_Subset(Dataset):

    def __init__(self, X, transform):
        self.b, _, _, _, self.t = X.shape
        self.X = rearrange(X, 'b c h w t -> (b t) c h w')
        

        B = self.b * self.t 
        self.indices = [i for i in range(B)]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.X[index, ...]

        xi, xj = self.transform(x)

        return xi, xj

class SimCLRDataTransform(object):
    def __init__(self, img_size=16, s = 1, kernel_size = 5): 
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size
        self.transform = self.get_simclr_pipeline_transform()

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = transforms.ToPILImage()(sample)

        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

    def get_simclr_pipeline_transform(self):

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.kernel_size)], p=0.8),
            transforms.ToTensor()
        ])

        return data_transforms

def get_simclr_data_loader(X, batch_size=128, num_workers=8):
    transform = SimCLRDataTransform()
    dataset = Sentinel_Subset(X, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    return data_loader