import argparse
import yaml
import sys
import os
import torch
import numpy as np
from liso import LISO
from diffusionstego import DiffusionSteg
from datasets.data_utils import get_dataset
from configs.paths_config import DATASET_PATHS


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path of the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--t_0', type=int, default=400, help='Num steps in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative process for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative process for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative process for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls the variance of the generative process')
    parser.add_argument('--lr_lat_opt', type=float, default=2e-6, help='Initial learning rate for latent optim')
    parser.add_argument('--bs_train', type=int, default=1, help='Training batch size')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--bpp', type=int, default=3, help='Payload: number of bits to be hidden per pixel')
    parser.add_argument('--dataset-class', type=str, default="AFHQ", help="Class of dataset")
    parser.add_argument('--num-images', type=int, default=200, help='Number of images to be optimized')
 
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def create_folders(args):
    #Define paths
    cover_path = os.path.join("Cover_images", args.dataset_class, str(args.bpp) + "bpp")
    stego_path = os.path.join("Stego_images", args.dataset_class, str(args.bpp) + "bpp")

    # Create directories if they don't exist
    os.makedirs(cover_path, exist_ok=True)
    os.makedirs(stego_path, exist_ok=True)

    print(f"Folders created: {cover_path}, {stego_path}")
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



def main():
    args, config = parse_args_and_config()

    # ----------- Folders to store images -----------#
    create_folders(args)

    # ----------- Model Stego -----------#
    model_steg = LISO.load(path=f"logs/{args.dataset_class}/{args.bpp}bpp/checkpoints/best.steg")
    
    model_steg.encoder.step_size = 1.0
    model_steg.mse_weight = 1.0
    model_steg.encoder.constraint = None

    
    

    # ----------- Data -----------#
    train_dataset, test_dataset = get_dataset(config.data.dataset, DATASET_PATHS, config)
        
    runner = DiffusionSteg(args, config)
    
    for index in range(args.num_images):
        random_image = train_dataset[index]
        random_image = random_image.unsqueeze(0)
        runner.finetune_steg_latent(model_steg, random_image, index, args.dataset_class, args.bpp)
  
    return 0


if __name__ == '__main__':
    sys.exit(main())
