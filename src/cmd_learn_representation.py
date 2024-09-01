# Learning a representation. For the time being, this code is learning a convolutional VAE.

import settings
import sys
sys.path.append(settings.CONV_VAE_DIR)

import argparse
import numpy as np
import torch
from tqdm import tqdm

# these imports are from the Conv-VAE-Torch package
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
# end of Conv-VAE-Torch imports

from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
#import os
import matplotlib.pyplot as plt
import argparse
import logging
#import socket



def get_config(): 
    """Conv-VAE: Creates the configuration. It is doing this by emulating the command line."""
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    value = ["this-script", f"-c{settings.VAE_CONFIG}"]
    # we are changing the parameters from here, to avoid changing the github 
    # downloaded package
    sys.argv = value
    config = ConfigParser.from_args(args)
    print(config)
    return config


def train(config):
    """Train the model"""
    # this would require a logger_config.json file
    # logger = config.get_logger('train')
    # let us just create a simple logger here
    logger = logging.getLogger(__name__)


    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      lr_scheduler=lr_scheduler)

    trainer.train()

def main():
    print("Learn a representation (the vision component)")
    config = get_config()
    train(config)

if __name__ == "__main__":
    main()

