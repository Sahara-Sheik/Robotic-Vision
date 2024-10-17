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

from encoding_conv_vae.conv_vae import get_config, create_configured_vae_json, latest_model, latest_training_run
from .sensor_processing import AbstractSensorProcessing

def latest_json_and_model(values):
    """Returns the latest Conv-Vae path and model, taking the information from the values dict of the config"""
    model_path = pathlib.Path(Config().values["conv_vae"]["model_dir"])
    model_path = pathlib.Path(model_path, "models", Config().values["conv_vae"]["model_name"])
    latest = latest_training_run(model_path)
    # print(latest)
    model_path = pathlib.Path(model_path, latest)
    model = latest_model(model_path)
    # The model from which we are starting        
    resume_model = pathlib.Path(model_path, model)
    jsonfile = pathlib.Path(model_path, "config.json")
    print(f"resume_model and jsonfile are:\n\tresume_model={resume_model}\n\tjsonfile={jsonfile}")
    return jsonfile, resume_model 

def get_conv_vae_config(jsonfile, resume_model, inference_only = True):
    """Returns the configuration object of the Experiment-Conv-Vae"""
    # As the code is highly dependent on the command line, emulating it here
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')


    # value = ["this-script", f"-c{file}", f"-r{model}"]
    value = ["this-script", f"-c{jsonfile}", f"-r{resume_model}"]

    # we are changing the parameters from here, to avoid changing the github downloaded package
    savedargv = sys.argv
    sys.argv = value
    config = ConfigParser.from_args(args)
    sys.argv = savedargv
    print(json.dumps(config.config, indent=4))
    # if it is inference only, remove the superfluously created directories.
    if inference_only:
        remove_dir = pathlib.Path(jsonfile.parent.parent, latest_training_run(jsonfile.parent.parent))
        remove_json = pathlib.Path(remove_dir, "config.json")
        print(f"Removing unnecessarily created json file: {remove_json.absolute()}")
        remove_json.unlink()
        print(f"Removing unnecessarily created package directory: {remove_dir.absolute()}")
        remove_dir.rmdir()
    return config


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

       