import numpy as np
from pathlib import Path
import cv2
import time

class CameraController:
    """This class is used to connect to a specified set of cameras. Images are captured from the cameras whenever the update is called. At the same time, it can also save the images to a specific directory and a specific name.
    
    # camera0 - webcam on the computer
    # camera2 - right mounted
    # camera3 - the free floating one
    # camera4 - the center mounted one 
    # cameras = [0, 2, 3, 4]
    # cameras = [0, 2]
    cameras = [4]    
    """
    def __init__(self, devices = [2, 3, 4], img_size = (128, 128)):
        """
        cameras: a list of numbers which correspond to the capture devices that will be captured
        dimension: the dimension to which the images are scaled down
        """
        self.img_size = img_size
        # create the capture devices
        self.capture_devs = {}
        for i in devices:
            cap = cv2.VideoCapture(i) 
            if cap is None or not cap.isOpened():
                print(f"Warning: unable to open video source: {i}")
            else:
                self.capture_devs[f"dev{i}"] = cap
                print(f"cap{i} works")
        self.caption = f"Cameras: {self.capture_devs.keys()}"
        self.images = {}
        self.visualize = True # if true, visualizes the captured images

    def stop(self):
        """When everything done, release the capture devices and close the windows"""
        for cap in self.capture_devs:
            self.capture_devs[cap].release()
        cv2.destroyAllWindows()

    def update(self):
        """
        Takes captures from all the active cameras, processes them, updates the window and optionally saves the images. Returns the key returned by waitKey()

        This one works, but it breaks down as soon as we have too many cameras
        """
        for index in self.capture_devs:
            cap = self.capture_devs[index]
            success, image = cap.read()
            if not success:
                continue
            if self.img_size != None:
                image = cv2.resize(image, self.img_size)
            self.images[index] = image
        # create a list of concatenated images
        imglist = list(self.images.values())
        concatenated_image = cv2.hconcat(imglist)
        try:
            if self.visualize:
                cv2.imshow(self.caption, concatenated_image)
                key = cv2.waitKey(1)
                return key
        except:
            print("Error at visualization? ")
