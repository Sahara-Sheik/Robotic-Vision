"""
Intermediate stuff: I am going to convert this into using a yaml file
----------------- old text -----------------------
This file contains settings for the VisionBasedRobotManipulator. These can be considered as constants for a particular run, but they might have different values on different computers or setups.

Should access them through settings.NAME
"""

import socket
import yaml
from pathlib import Path

class Config:
    """The overall settings class. """
    _instance = None  # Class-level attribute to store the instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            home_directory = Path.home()
            main_config = Path(home_directory, ".config", "VisionBasedRobotManipulator","mainsettings.yaml")
            if not main_config.exists():
                raise Exception("Missing main config file: {main_config}")
            with main_config.open("rt") as handle:
                main_config = yaml.safe_load(handle)
            configpath = main_config["configpath"]
            print("Proceeding to load config file: {configpath}")
            configpath = Path(configpath)
            if not configpath.exists():
                raise Exception("Missing config file: {configpath}")
            with configpath.open("rt") as handle:
                cls._instance.values = yaml.safe_load(handle)
        return cls._instance

# setting the specific directories based on the current machine
machine_name = socket.gethostname()
print("The name of the current machine is:" + machine_name)

if (machine_name == "LotziYoga"):
    Config().load_settings("/home/lboloni/Documents/Hackingwork/_Checkouts/VisionBasedRobotManipulator/src/settings-tredy2.yaml")

if (machine_name == "tredy"):
    Config().load_settings("/home/lboloni/Insync/lotzi.boloni@gmail.com/Google Drive/LotziStudy/Code/PackageTracking/VisionBasedRobotManipulator/settings/settings-tredy2.yaml")



# Definitions of the machine dependent settings

# The USB port where the AL5D robot is connected on this particular computer.
# alternatives: linux: /dev/ttyUSB0, /dev/ttyUSB1
ROBOT_USB_PORT = "/dev/ttyUSB0" 

# The camera numbers which should be used for the robot, as a list
# Typically 0 is the webcam of the computer, other ones in the order they are connected
ACTIVE_CAMERA_LIST = [2] 

# Data directory for demonstrations. This is where demonstrations are saved, or loaded from in the case of replay
DEMO_DIR = "."

# Data directory for training data. This is where the training data is kept and organized 
TRAINING_DATA_DIR = "."

# Directory for models and checkpoints
MODEL_DIR = "."

# Directory for the checkout of the CONV_VAE project 
# https://github.com/julian-8897/Conv-VAE-PyTorch
CONV_VAE_DIR = "."

## Size of the latent encoding
## FIXME: in the case of the Conv-VAE, this number must be the same as the one
## in the conf-robot-machinename.json, latent_dims
LATENT_ENCODING_SIZE = 128

# setting the specific directories based on the current machine
machine_name = socket.gethostname()
print("The name of the current machine is:" + machine_name)

if (machine_name == "LotziYoga"):
    ROBOT_USB_PORT = "USB1" # Fixme, windows usb port access 
    ACTIVE_CAMERA_LIST = [2] 
    # VAE configuration
    VAE_CONFIG = "config-robot-lotziyoga.json"
    VAE_MODEL_DIR = "." # FIXME, specify it
    VAE_TRAININGDATA_DIR = "." # FIXME, specify it

if (machine_name == "tredy2"):
    ROBOT_USB_PORT = "/dev/ttyUSB0" 
    ACTIVE_CAMERA_LIST = [2] 
    DEMO_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-demos"
    TRAINING_DATA_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-training-data"
    MODEL_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-models"
    # VAE configuration
    CONV_VAE_DIR = "/home/lboloni/Documents/Hackingwork/_Checkouts/Julian-8897-Conv-VAE-PyTorch/Conv-VAE-PyTorch"
    VAE_CONFIG = "/home/lboloni/Documents/Hackingwork/_Checkouts/VisionBasedRobotManipulator/src/encoding-conv-vae/config-robot-tredy2.json"
    VAE_MODEL_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-models/Conv-VAE/"
    VAE_MODEL_FILE = "0817_155635/checkpoint-epoch20.pth"
    VAE_TRAININGDATA_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-training-data/vae-training-data" 

