import sys
sys.path.append("..")
import settings

import numpy as np

class AbstractSensorProcessing:

    def process(self, sensor_readings):
        """Processes the sensor input and returns the latent encoding, which is a vector of the size of the latent encoding"""
        return np.zeros(settings.LATENT_ENCODING_SIZE)
    
print("Hello world")