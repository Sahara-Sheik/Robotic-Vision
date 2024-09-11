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
                raise Exception(f"Missing main config file: {main_config}")
            with main_config.open("rt") as handle:
                main_config = yaml.safe_load(handle)
            configpath = main_config["configpath"]
            print(f"Proceeding to load config file: {configpath}")
            configpath = Path(configpath)
            if not configpath.exists():
                raise Exception(f"Missing config file: {configpath}")
            with configpath.open("rt") as handle:
                cls._instance.values = yaml.safe_load(handle)
        return cls._instance

