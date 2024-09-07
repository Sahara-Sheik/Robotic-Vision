from .sensor_processing import AbstractSensorProcessing

import sys
sys.path.append("..")
import settings
sys.path.append(settings.CONV_VAE_DIR)

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

from mpl_toolkits.axes_grid1 import ImageGrid


class VaeSensorProcessing:
    """Sensor processing based on a pre-trained Conv-VAE"""