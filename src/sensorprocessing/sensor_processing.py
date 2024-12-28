import sys
sys.path.append("..")
from settings import Config

import numpy as np
from .sp_helper import load_picturefile_to_tensor


class AbstractSensorProcessing:
    """The ancestor of all the classes that perform a sensor processing"""

    def process(self, sensor_readings):
        """Processes the sensor input and returns the latent encoding, which is a vector of the size of the latent encoding. 
        This is intended to be used during real-time deployment"""
        return np.zeros(Config().values["robot"]["latent_encoding_size"])

    #def process_file(self, sensor_readings_file):
    #    """Assumes that the sensor readings are in a file (eg. a picture). The format of the file is up to the particular processing, and it might be a directory to a miscellaneous set of readings. Processes the sensor input and returns the latent encoding, which is a vector of the size of the latent encoding. 
    #    This is intended to be used during training. """
    #    return np.zeros(Config().values["robot"]["latent_encoding_size"])

    def process_file(self, sensor_readings_file):
        """Processsed file"""
        sensor_readings, image = load_picturefile_to_tensor(sensor_readings_file, self.transform)
        output = self.process(sensor_readings)
        return output
