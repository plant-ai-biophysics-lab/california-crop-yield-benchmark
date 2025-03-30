import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from src.configs import set_seed, Configs
set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

from src.dataLoader import dataloader
from utils import misc
import argparse


def main(args):

    data_loader_training, data_loader_validate, data_loader_test = dataloader(
    county_names = ['Monterey', 'Yolo', 'Merced', 'Fresno', 'Imperial'], batch_size = args.batch_size
    )

    config = Configs(
        embed_dim =  args.embed_dim, 
        landsat_channels =  args.landsat_channels, 
        et_channels =  args.et_channels, 
        climate_variables =  args.climate_variables, 
        soil_variables =  args.soil_variables, 
        num_heads =  args.num_heads, 
        num_layers =  args.num_layers,
        attn_dropout =  args.attn_dropout, 
        proj_dropout =  args.proj_dropout, 
        timeseries =  args.timeseries,
        pool =  False
    ).call()
    

    M = misc.FineTune(exp_name =  args.exp_name, device =  device)
    yieldbenchmark = M.fit(config)

    # M.train(
    #     model = yieldbenchmark, 
    #     dataloader_train = data_loader_training, 
    #     dataloader_valid = data_loader_validate,
    #     optimizer =  args.optimizer, 
    #     loss =  args.loss, 
    #     lr =  args.lr,
    #     wd =  args.wd, 
    #     epochs=  args.epochs)

    M.predict(model = yieldbenchmark, 
              data_loader = data_loader_training, 
              category = 'train')
    
    M.predict(model = yieldbenchmark, 
              data_loader = data_loader_validate, 
              category = 'valid')
    
    M.predict(model = yieldbenchmark, 
              data_loader = data_loader_test, 
              category = 'test')

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Imbalance Deep Yield Estimation")
    parser.add_argument("--exp_name", type=str, default = "Mar23", help = "Experiment name")
    parser.add_argument("--batch_size", type=int, default = 2, help = "Batch size")
    parser.add_argument("--embed_dim", type=int, default = 512, help = "Embedding Dimension")
    parser.add_argument("--landsat_channels", type=int, default = 6, help = "The number of landsat channels")
    parser.add_argument("--et_channels", type=int, default = 2, help = "The number of et channels")
    parser.add_argument("--climate_variables", type=int, default = 8, help = "The number of climate variables")
    parser.add_argument("--soil_variables", type=int, default = 5, help = "The number of soil variables")
    parser.add_argument("--num_heads", type=int, default = 8, help = "The number of heads")
    parser.add_argument("--num_layers", type=int, default = 6, help = "The number of Layers of attention")
    parser.add_argument("--attn_dropout", type=float, default = 0.1, help = "Attention dropout")
    parser.add_argument("--proj_dropout", type=float, default = 0.1, help = "Projecttion dropout")
    parser.add_argument("--timeseries", type=str, default = True, help = "Timeseries defult")
    parser.add_argument("--lr", type=float, default = 0.0001, help = "Learning rate")
    parser.add_argument("--wd", type=float, default = 0.01, help = "Value of weight decay")
    parser.add_argument("--epochs", type=int, default = 150, help = "The number of epochs")
    parser.add_argument("--loss", type=str, default = "mse", help = "Loss function")
    parser.add_argument("--optimizer", type=str, default = "adamw", help = "Optimizer")


    args = parser.parse_args()

    main(args)