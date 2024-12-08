from .sensor_processing import AbstractSensorProcessing

import sys
sys.path.append("..")

from settings import Config
sys.path.append(Config().values["conv_vae"]["code_dir"])

#import argparse
import numpy as np
import torch
from tqdm import tqdm

from PIL import Image


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
#import os
import matplotlib.pyplot as plt
import argparse
#import socket
#import pathlib
#import json

from mpl_toolkits.axes_grid1 import ImageGrid

from encoding_conv_vae.conv_vae import get_config, create_configured_vae_json, latest_model, latest_training_run, latest_json_and_model, get_conv_vae_config, get_transform_to_robot

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import load_picturefile_to_tensor


class ConvVaeSensorProcessing (AbstractSensorProcessing):
    """Sensor processing based on a pre-trained Conv-VAE"""

    def __init__(self, conv_vae_jsonfile, resume_model_pthfile):
        """Restore a pre-trained model based on the configuration json file and the model file"""
        self.conv_vae_jsonfile = conv_vae_jsonfile
        self.resume_model_pthfile = resume_model_pthfile
        self.vae_config = get_conv_vae_config(
            self.conv_vae_jsonfile, 
            self.resume_model_pthfile, 
            inference_only=True)
        # build model architecture
        self.model = self.vae_config.init_obj('arch', module_arch)
        self.loss_fn = getattr(module_loss, self.vae_config['loss'])
        # loading the checkpoint
        self.checkpoint = torch.load(self.vae_config.resume, map_location=torch.device('cpu'))
        self.state_dict = self.checkpoint['state_dict']
        self.model.load_state_dict(self.state_dict)
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = get_transform_to_robot()


    def __init__old(self):
        """
        TODO: Once it is cleaned up, transfer this code back to encoding-conv-vae and use it in Experiment-Conv-Vae as well to
        load the latest model, and use it to process
        """
        # FIXME: in the Experiment-CONV-VAE use these... 
        self.jsonfile, self.resume_model = latest_json_and_model(Config().values)
        self.config = get_conv_vae_config(self.jsonfile, self.resume_model, inference_only=True)
        # build model architecture
        self.model = self.config.init_obj('arch', module_arch)
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
        self.transform = get_transform_to_robot()

    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor. 
        Returns the z encoding in the form of a numpy array."""
        with torch.no_grad():
            output, mu, logvar = self.model(sensor_readings)
        mus = torch.squeeze(mu)
        return mus.cpu().numpy()
    
    def process_file(self, sensor_readings_file):
        """"""
        sensor_readings, image = load_picturefile_to_tensor(sensor_readings_file, self.transform)
        output = self.process(sensor_readings)
        return output
        
        