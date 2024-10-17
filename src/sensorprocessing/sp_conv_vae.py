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
import json

from mpl_toolkits.axes_grid1 import ImageGrid

from encoding_conv_vae.conv_vae import get_config, create_configured_vae_json, latest_model, latest_training_run, latest_json_and_model, get_conv_vae_config, load_image_to_tensor, get_transform

from .sensor_processing import AbstractSensorProcessing

class ConvVaeSensorProcessing (AbstractSensorProcessing):
    """Sensor processing based on a pre-trained Conv-VAE"""

    def __init__(self):
        """
        TODO: Once it is cleaned up, transfer this code back to encoding-conv-vae and use it in Experiment-Conv-Vae as well to
        load the latest model, and use it to process
        """
        self.jsonfile, self.resume_model = latest_json_and_model(Config().values)
        self.config = get_conv_vae_config(self.jsonfile, self.resume_model)

        self.data_loader = getattr(module_data, self.config['data_loader']['type'])(
            self.config['data_loader']['args']['data_dir'],
            batch_size=36,
            shuffle=False,
            validation_split=0.0,
            # training=False,
            num_workers=2
        )

        # LOTZI: this is an uninitialized model architecture
        # build model architecture
        self.model = self.config.init_obj('arch', module_arch)
        # logger.info(self.model)
        print(self.model)

        # get function handles of loss and metrics
        self.loss_fn = getattr(module_loss, self.config['loss'])
        # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        # logger.info('Loading checkpoint: {} ...'.format(config.resume))

        # loading on CPU-only machine
        print("Loading the checkpoint")
        self.checkpoint = torch.load(self.config.resume, map_location=torch.device('cpu'))
        print("Checkpoint loaded")

        self.state_dict = self.checkpoint['state_dict']
        if self.config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(self.state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = get_transform()

    def process(self, sensor_readings):
        """Let us assume that the sensor readings are in a file"""
        input, image = load_image_to_tensor(sensor_readings, self.transform)
        output, mu, logvar = self.model(input)
        return mu