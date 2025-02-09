"""
sp_cnn.py

Sensor processing using pretrained CNN
"""
import sys
sys.path.append("..")
from settings import Config

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_robot, load_picturefile_to_tensor

import pathlib
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class VGG19Regression(nn.Module):
    """Neural network used to create a latent embedding. Starts with a VGG19 neural network, without the classification head. The features are flattened, and fed into a regression MLP trained on visual proprioception. 
    
    When used for encoding, the processing happens only to an internal layer in the MLP.
    """

    def __init__(self, hidden_size, output_size):

        super(VGG19Regression, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = vgg19.features
        self.flatten = nn.Flatten()  # Flatten the output for the fully connected layer
        self.model = nn.Sequential(
            # The internal size seem to depend on the external size. 
            # the original with 7 * 7 corresponded to the 224 x 224 inputs
            #nn.Linear(512 * 7 * 7, hidden_size),
            nn.Linear(512 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False        

    def forward(self, x):
        """Forward the input image through the complete network for the purposes of training using the proprioception. Return a vector of output_size which """
        features = self.feature_extractor(x)
        #print(features.shape)
        flatfeatures = self.flatten(features)
        #print(flatfeatures.shape)
        output = self.model(flatfeatures)
        # print(output.device)
        return output
    
    def encode(self, x):
        """Performs an encoding of the input image, by forwarding though the encoding and first three layers."""
        features = self.feature_extractor(x)
        flatfeatures = self.flatten(features)
        h1 = self.model[0](flatfeatures)
        h2 = self.model[1](h1)
        h3 = self.model[2](h2)
        return h3
        

class VGG19SensorProcessing(AbstractSensorProcessing):
    """Sensor processing using a pre-trained VGG19 architecture from above."""

    def __init__(self, exp, device="cpu"):
        """Create the sensormodel """
        # self.transform = get_transform_to_robot()
        # FIXME: I think that this was trained on different size...
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.exp = exp
        hidden_size = exp["latent_dims"]
        output_size = Config()["robot"]["action_space_size"]
        self.enc = VGG19Regression(hidden_size=hidden_size, output_size=output_size)
        self.enc = self.enc.to(device)
        modelfile = pathlib.Path(exp["data_dir"], 
                                exp["proprioception_mlp_model_file"])
        assert modelfile.exists()
        self.enc.load_state_dict(torch.load(modelfile))

    def process(self, sensor_readings):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor. 
        Returns the z encoding in the form of a numpy array."""
        print(f"sensor readings shape {sensor_readings.shape}")
        with torch.no_grad():
            z = self.enc.encode(sensor_readings)
        z = torch.squeeze(z)
        return z.cpu().numpy()
    
    def process_file(self, sensor_readings_file):
        """Processsed file"""
        sensor_readings, image = load_picturefile_to_tensor(sensor_readings_file, self.transform)
        output = self.process(sensor_readings)
        return output
    
        