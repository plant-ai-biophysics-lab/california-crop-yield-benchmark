import torch
import torch.nn as nn
import sys
from einops import rearrange
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np


# from src.configs import set_seed 
# set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def plot_scatter_subplots(train_df, valid_df, test_df, month):
    """
    Creates scatter plots for train, valid, and test datasets in a single row with three columns.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        valid_df (pd.DataFrame): Validation dataset.
        test_df (pd.DataFrame): Test dataset.
        month (int): The month number corresponding to the ypred_m column (e.g., 12 for ypred_m12).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    datasets = {'Train': train_df, 'Valid': valid_df, 'Test': test_df}
    
    # Determine global axis limits
    all_true = pd.concat([train_df['ytrue'], valid_df['ytrue'], test_df['ytrue']])
    all_pred = pd.concat([train_df[f"ypred_m{month}"], valid_df[f"ypred_m{month}"], test_df[f"ypred_m{month}"]])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    buffer = (max_val - min_val) * 0.05
    axis_min, axis_max = 0 , 60 

    for i, (ax, (name, df)) in enumerate(zip(axes, datasets.items())):
        ypred_col = f"ypred_m{month}"

        # Compute evaluation metrics
        r2 = r2_score(df["ytrue"], df[ypred_col])
        rmse = np.sqrt(mean_squared_error(df["ytrue"], df[ypred_col]))
        mae = mean_absolute_error(df["ytrue"], df[ypred_col])
        
        # Scatter plot
        sns.scatterplot(
            data=df, x="ytrue", y=ypred_col, hue="crop_name",
            palette="tab10", legend=False, s=50, marker='o', ax=ax
        )

        # Fit and plot regression line
        model = LinearRegression().fit(df[["ytrue"]], df[ypred_col])
        x_vals = np.array([axis_min, axis_max])
        y_vals = model.predict(x_vals.reshape(-1, 1))
        ax.plot(x_vals, y_vals, color='black', linestyle='--', label='Fit Line')

        # Add metrics text
        metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.7", edgecolor="black", facecolor="white"))

        # Axis labels and title
        ax.set_title(f"{name} Scatter Plot")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        # if i == 0:
        #     
        # else:
        #     ax.set_ylabel("")  # Hide y-axis label only, not ticks
        ax.set_ylabel("Predicted Yield (ton/ac)")
        ax.set_xlabel("Observed Yield (ton/ac)")
        ax.grid(False)

    plt.tight_layout()
    plt.show()




def plot_monthly_metrics(train_df, valid_df, test_df):
    def compute_metrics(df):
        r2s, rmses, maes = [], [], []
        for i in range(1, 13):
            y_true = df['ytrue']
            y_pred = df[f'ypred_m{i}']
            r2s.append(r2_score(y_true, y_pred))
            rmses.append(mean_squared_error(y_true, y_pred, squared=False))
            maes.append(mean_absolute_error(y_true, y_pred))
        return r2s, rmses, maes

    # Compute metrics
    train_r2, train_rmse, train_mae = compute_metrics(train_df)
    valid_r2, valid_rmse, valid_mae = compute_metrics(valid_df)
    test_r2, test_rmse, test_mae = compute_metrics(test_df)

    months = list(range(1, 13))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

    # R²
    axs[0].plot(months, train_r2, marker='o', label='Train', color='blue')
    axs[0].plot(months, valid_r2, marker='o', label='Validation', color='orange')
    axs[0].plot(months, test_r2, marker='o', label='Test', color='green')
    axs[0].set_title('R² per Month')
    axs[0].set_xlabel('Month')
    axs[0].set_ylabel('R²')
    axs[0].set_xticks(months)
    axs[0].set_ylim(0, 1)
    axs[0].legend()
    axs[0].grid(False)

    # RMSE
    axs[1].plot(months, train_rmse, marker='o', label='Train', color='blue')
    axs[1].plot(months, valid_rmse, marker='o', label='Validation', color='orange')
    axs[1].plot(months, test_rmse, marker='o', label='Test', color='green')
    axs[1].set_title('RMSE per Month')
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('RMSE')
    axs[1].set_xticks(months)
    axs[1].set_ylim(20, 50)
    axs[1].legend()
    axs[1].grid(False)

    # MAE
    axs[2].plot(months, train_mae, marker='o', label='Train', color='blue')
    axs[2].plot(months, valid_mae, marker='o', label='Validation', color='orange')
    axs[2].plot(months, test_mae, marker='o', label='Test', color='green')
    axs[2].set_title('MAE per Month')
    axs[2].set_xlabel('Month')
    axs[2].set_ylabel('MAE')
    axs[2].set_xticks(months)
    axs[2].set_ylim(20, 50)
    axs[2].legend()
    axs[2].grid(False)

    plt.tight_layout()
    plt.show()

def evaluate_by_crop(df):
    results = []

    for crop in df['crop_name'].unique():
        group = df[df['crop_name'] == crop]
        y_true = group['ytrue']
        y_pred = group['ypred_m12']

        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            'crop_name': crop,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

    return pd.DataFrame(results)



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
    

def finetune_train(model, criterion, optimizer, dataloader, epochs):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        loss_train_value = 0
        top1_train_accuracy, top5_train_accuracy = 0, 0
        for counter, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch[..., -1].to(device)
            # x_batch = rearrange(x_batch, 'b c h w t -> (b t) c h w')
            y_batch = y_batch.to(device)

            # Forward pass
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_train_accuracy += top1[0]
            top5_train_accuracy += top5[0]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_value += loss.item()


        top1_train_accuracy /= (counter + 1)
        top5_train_accuracy /= (counter + 1)
        
        print(f"Epoch {epoch}: cross entropy loss = {(loss_train_value /len(dataloader)):.4f} \tTop1 Train accuracy {top1_train_accuracy.item()}\tTop5 Train acc: {top5_train_accuracy.item()}")

def extract_features(model, dataloader):
    model.eval()  # Set model to evaluation mode
    features = []
    labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch[..., -1].to(device)
            # Get the 128-dimensional representation from the base model
            feature_batch = model.base_model(x_batch)
            features.append(feature_batch.cpu().numpy())
            labels.append(y_batch.cpu().numpy())

            
    # Concatenate all batches
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def extract_features2(model, dataloader):
    model.eval()  # Set model to evaluation mode
    features = []
    labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch[..., -1].to(device)
            # Get the 128-dimensional representation from the base model
            feature_batch = model(x_batch)
            features.append(feature_batch.cpu().numpy())
            labels.append(y_batch.cpu().numpy())
    # Concatenate all batches
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def get_predictions_and_labels(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch[..., -1].to(device)
            y_batch = y_batch.to(device)
            # Get predictions from the model
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            
    return np.array(all_preds), np.array(all_labels)