from .sensor_processing import AbstractSensorProcessing

import sys
sys.path.append("..")

from settings import Config
sys.path.append(Config().values["conv_vae"]["code_dir"])

import argparse
import numpy as np
import torch
from tqdm import tqdm

# these imports are from the Conv-VAE package
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import argparse
import socket
import pathlib

from mpl_toolkits.axes_grid1 import ImageGrid

from encoding_conv_vae.conv_vae import get_config, create_configured_vae_json, latest_model, latest_training_run
from .sensor_processing import AbstractSensorProcessing

class ConvVaeSensorProcessing (AbstractSensorProcessing):
    """Sensor processing based on a pre-trained Conv-VAE"""

    def __init__(self):
        """To do: basically, the code here be similar to the one in Experiment-Conv-Vae - load the latest model, and use it to process
        """
        model_path = pathlib.Path(Config().values["conv_vae"]["model_dir"])
        model_path = pathlib.Path(model_path, "models", Config().values["conv_vae"]["model_name"])
        latest = latest_training_run(model_path)
        print(latest)
        model_path = pathlib.Path(model_path, latest)
        model = latest_model(model_path)
        print(model_path)
        print(model)      

        # As the code is highly dependent on the command line, emulating it here
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
        
        resume_model = pathlib.Path(model_path, model)
        jsonfile = pathlib.Path(model_path, "config.json")


        # value = ["this-script", f"-c{file}", f"-r{model}"]
        value = ["this-script", f"-c{jsonfile}", f"-r{resume_model}"]

        # we are changing the parameters from here, to avoid changing the github 
        # downloaded package
        sys.argv = value
        config = ConfigParser.from_args(args)
        print(config)