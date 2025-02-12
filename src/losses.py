import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import time
from libpysal.weights import lat2W
from esda.moran import Moran
import numpy as np
from multiprocessing import Pool
from models.configs import set_seed
set_seed(1987)


device = "cuda" if torch.cuda.is_available() else "cpu"


class MoranCalculator(_Loss):
    def __init__(self, n = 16):
        super(MoranCalculator, self).__init__()
        """
        Initialize the MoranCalculator with the number of elements in the data array.

        Parameters:
        - n (int): Number of elements in the data array.
        """
        self.n = n

    def morans_i(self, data, w):
        """
        Calculate Moran's I for a given data array and spatial weights matrix.

        Parameters:
        - data (torch.Tensor): Input data array (1D tensor).
        - w (torch.Tensor): Spatial weights matrix (2D tensor).

        Returns:
        - float: Moran's I value.
        """
        n = data.shape[0]
        mean_data = torch.mean(data)
        deviations = data - mean_data
        numerator = torch.sum(w * torch.outer(deviations, deviations))
        denominator = torch.sum(deviations**2)
        s = torch.sum(w)
        i_value = (n /s) * (numerator / denominator)

        return i_value.item()  # Convert to Python float

    def calculate_w(self):
        """
        Calculate a spatial weights matrix based on rook contiguity for a 2D array of data.

        Returns:
        - torch.Tensor: Spatial weights matrix (2D tensor).
        """
        w = torch.zeros(self.n * self.n, self.n * self.n)
        for i in range(self.n * self.n):
            for j in range(self.n * self.n):
                if abs(i // self.n - j // self.n) + abs(i % self.n - j % self.n) == 1:
                    w[i, j] = 1  # Assign a weight of 1 for rook contiguity

        return w

    def calculate_morans_i(self, inputs, targets, w):
        """
        Calculate the difference in Moran's I for two datasets.

        Parameters:
        - inputs (torch.Tensor): Input data array (2D tensor).
        - targets (torch.Tensor): Target data array (2D tensor).
        - w (torch.Tensor): Spatial weights matrix (2D tensor).

        Returns:
        - torch.Tensor: Difference in Moran's I.
        """
        mi_prediction = self.morans_i(inputs.view(-1), w)
        mi_target = self.morans_i(targets.view(-1), w)

        return (mi_prediction - mi_target)**2

    def forward(self, inputs, targets):
        """
        Calculate the mean squared error loss with Moran's I for a batch of inputs and targets.

        Parameters:
        - inputs (torch.Tensor): Batch of input data (4D tensor).
        - targets (torch.Tensor): Batch of target data (4D tensor).

        Returns:
        - torch.Tensor: Combined loss.
        """
        # Move inputs and targets to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        targets = targets.to(device)

        # MSE
        mse = torch.mean((inputs - targets) ** 2) 

        # MORANS I
        batch_size, _, height, width = inputs.shape
        w = self.calculate_w()
        w = w.to(device)
        morans_i_diffs = [self.calculate_morans_i(inputs[i, 0, ...], targets[i, 0, ...], w) for i in range(batch_size)]

        morans_i_loss = torch.mean(torch.tensor(morans_i_diffs, dtype=torch.float32))
        alpha = 50
        loss = mse + (morans_i_loss * alpha)

        return loss

def calculate_morans_i(inputs, targets, w):

    mi_prediction = Moran(inputs.reshape(-1), w)
    mi_target = Moran(targets.reshape(-1), w)
    return (mi_prediction.I - mi_target.I) ** 2

def mse_morans_i_loss(inputs, targets):
    # MSE
    mse = torch.mean((inputs - targets) ** 2)/ 10

    # MORANS I
    batch_size, _, height, width = inputs.shape
    w = lat2W(height, width)

    # Use multiprocessing for parallel processing
    with Pool() as pool:
        morans_i_diffs = pool.starmap(calculate_morans_i, [(inputs[i].detach().cpu().numpy(), targets[i].detach().cpu().numpy(), w) for i in range(batch_size)])

    morans_i_loss = np.mean(morans_i_diffs) 
    loss = mse + torch.tensor(morans_i_loss, dtype=torch.float32)

    return loss

def mse_loss(inputs, targets):
    loss = (inputs - targets) ** 2
    loss = torch.mean(loss)
    
    return loss

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

