"""
This file contains settings for the VisionBasedRobotManipulator. These can be considered as constants for a particular run, but they might have different values on different computers or setups.

Should access them through settings.NAME
"""

import socket

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

# setting the specific directories based on the current machine
machine_name = socket.gethostname()
print("The name of the current machine is:" + machine_name)

if (machine_name == "LotziYoga"):
    ROBOT_USB_PORT = "USB1" # Fixme, windows usb port access 
    ACTIVE_CAMERA_LIST = [2] 


if (machine_name == "tredy2"):
    ROBOT_USB_PORT = "/dev/ttyUSB0" 
    ACTIVE_CAMERA_LIST = [2] 
    DEMO_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-demos"
    TRAINING_DATA_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-training-data"
    MODEL_DIR = "/home/lboloni/Documents/Hackingwork/__Temporary/VisionBasedRobotManipulator-models"
    CONV_VAE_DIR = "/home/lboloni/Documents/Hackingwork/_Checkouts/Julian-8897-Conv-VAE-PyTorch/Conv-VAE-PyTorch"
