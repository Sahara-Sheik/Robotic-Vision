"""
sp_cnn.py

Sensor processing using pretrained CNN
"""
import sys
sys.path.append("..")

from .sensor_processing import AbstractSensorProcessing

import torch
import torch.nn as nn
from torchvision import models

class CNNSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using pretrained CNN, in particular ImageNet pretrained architectures"""

    def __init__(self, modelname, latent_size):
        """Create the models"""
        # Step 1: Load the pre-trained VGG-19 model
        if modelname == "vgg19":
            assert latent_size == 512
            vgg19 = models.vgg19(pretrained=True)
            vgg19.eval()  # Set the model to evaluation mode
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vgg19.to(device)
            self.model = vgg19.features



    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor. 
        Returns the z encoding in the form of a numpy array."""
        with torch.no_grad():
             = self.model(sensor_readings)
        mus = torch.squeeze(mu)
        return mus.cpu().numpy()
    
        