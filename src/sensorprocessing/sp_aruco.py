"""
sp_aruco.py

Sensor processing using Aruco markers
"""
import sys
sys.path.append("..")
from settings import Config

from .sensor_processing import AbstractSensorProcessing
from .sp_helper import get_transform_to_robot, load_picturefile_to_tensor

import numpy as np
import cv2


class ArucoSensorProcessing(AbstractSensorProcessing):
    """Sensor processing using a pre-trained VGG19 architecture from above."""

    def __init__(self, exp, device="cpu"):
        """Create the sensormodel """
        super().init(exp, device)
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.MARKER_COUNT = exp["MARKER_COUNT"]
        self.XMAX = exp["XMAX"]
        self.YMAX = exp["YMAX"]
        self.NORMALIZER = np.tile([self.XMAX, self.YMAX], 4)
        if self.latent_size < self.MARKER_COUNT * (8+1):
            raise Exception(f"Latent size {self.latent_size} too small for {self.MARKER_COUNT} markers!")


    def process(self, sensor_image):
        """Process a sensor readings object - in this case it must be an image prepared into a batch by load_image_to_tensor or load_capture_to_tensor."""
        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
                sensor_image, self.arucoDict, 
                parameters=self.arucoParams)

        z = np.ones(self.latent_size) * -1.0
        for id, corners in zip(marker_ids, marker_corners):
            detection = corners[0].flatten() / self.NORMALIZER
            idn = id[0]
            z[idn * (8+1):(idn+1) * (8+1)-2] = detection
            z[(idn+1) * (8+1)-1] = 1.0 # mark the fact that it is present
        return z

