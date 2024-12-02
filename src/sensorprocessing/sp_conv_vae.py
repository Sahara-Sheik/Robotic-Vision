from .sensor_processing import AbstractSensorProcessing

import sys
sys.path.append("..")

from settings import Config
sys.path.append(Config().values["conv_vae"]["code_dir"])

import argparse
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
import os
import matplotlib.pyplot as plt
import argparse
import socket
import pathlib
import json

from mpl_toolkits.axes_grid1 import ImageGrid

from encoding_conv_vae.conv_vae import get_config, create_configured_vae_json, latest_model, latest_training_run, latest_json_and_model, get_conv_vae_config, get_transform

from .sensor_processing import AbstractSensorProcessing


def load_image_to_tensor(picture_file, transform):
    """Loads an image from a file and transforms it into a single 
    element batch. Returns the batch and the image in a displayable format

    FIXME: this needs to be cleaned up, it was made very quickly for the
    VAE experiments

    """
    # Load an image using PIL
    image = Image.open(picture_file)
    # print(image.mode)
    # FIXME: if this is already RGB, this is not needed
    # at least for the medical image, this is in 16 bit unsigned integer
    image_rgb = image.convert("RGB")
    # transform, scale, convert to tensor
    image_tensor = transform(image_rgb)
    # Display some information about the image tensor
    # print(image_tensor.shape)  # e.g., torch.Size([3, H, W])
    # Convert the tensor to a format suitable for matplotlib (from [C, H, W] to [H, W, C])
    image_tensor_for_pic = image_tensor.permute(1, 2, 0)
    #plt.imshow(image_tensor_for_pic)
    # Add a batch dimension: shape becomes [1, 3, 224, 224]
    image_batch = image_tensor.unsqueeze(0)

    # Move tensor to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_batch_device = image_batch.to(device)
    return image_batch_device, image_tensor_for_pic


def load_capture_to_tensor(image_from_camera, transform):
    """Gets an image as returned by the camera controller and transforms it into a single 
    element batch. Returns the batch and the image in a displayable format

    FIXME: this needs to be cleaned up, it was collected from the VAE experiments
    
    """
    # print(image.mode)
    # FIXME: if this is already RGB, this is not needed
    # at least for the medical image, this is in 16 bit unsigned integer
    # image_rgb = image.convert("RGB")
    image = Image.fromarray(image_from_camera)
    image_rgb = image.convert("RGB")
    # transform, scale, convert to tensor
    image_tensor = transform(image_rgb)
    # Display some information about the image tensor
    # print(image_tensor.shape)  # e.g., torch.Size([3, H, W])
    # Convert the tensor to a format suitable for matplotlib (from [C, H, W] to [H, W, C])
    image_tensor_for_pic = image_tensor.permute(1, 2, 0)
    #plt.imshow(image_tensor_for_pic)
    # Add a batch dimension: shape becomes [1, 3, 224, 224]
    image_batch = image_tensor.unsqueeze(0)

    # Move tensor to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_batch_device = image_batch.to(device)
    return image_batch_device, image_tensor_for_pic



class ConvVaeSensorProcessing (AbstractSensorProcessing):
    """Sensor processing based on a pre-trained Conv-VAE"""

    def __init__(self):
        """
        TODO: Once it is cleaned up, transfer this code back to encoding-conv-vae and use it in Experiment-Conv-Vae as well to
        load the latest model, and use it to process
        """
        self.jsonfile, self.resume_model = latest_json_and_model(Config().values)
        self.config = get_conv_vae_config(self.jsonfile, self.resume_model, inference_only=True)

        # LOTZI: I don't think that we need the data loader here
        #self.data_loader = getattr(module_data, self.config['data_loader']['type'])(
        #    self.config['data_loader']['args']['data_dir'],
        #    batch_size=36,
        #    shuffle=False,
        #    validation_split=0.0,
            # training=False,
        #    num_workers=2
        #)

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
        with torch.no_grad():
            output, mu, logvar = self.model(sensor_readings)
        mus = torch.squeeze(mu)
        return mus.cpu().numpy()
    
    def process_file(self, sensor_readings_file):
        sensor_readings, image = load_image_to_tensor(sensor_readings_file, self.transform)
        output = self.process(sensor_readings)
        return output
        
        