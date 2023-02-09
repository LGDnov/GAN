from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import configargparse




def dataloader(dataroot, image_size, batch_size, workers = 2):
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader

#real_batch = next(iter(dataloader))
def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                    help='name project')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='number epochs')
    parser.add_argument('--config', is_config_file=True,
                    help='config file path')
    parser.add_argument("--path_model", type=str,
                        default='./models/', help='directory for models')
    parser.add_argument("--data_train", type=str,
                        default='./data/simple', help='input data directory')
    parser.add_argument("--image_size", type=int, default=64,
                        help='resize image')
    parser.add_argument("--batch_size", type=int, default=128,
                        help='batch size')
    parser.add_argument("--workers", type=int, default=2,
                        help='how many sub-processes to use for data loading')
    parser.add_argument("--load_model_path", type=str, default=None,
                        help='path model for load')
    parser.add_argument("--load_model_name", type=str, default=None,
                        help='name model for load')
    return parser