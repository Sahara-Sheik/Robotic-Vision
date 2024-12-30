"""
demo_to_trainingdata.py

Create training data from demonstrations. 
"""
import sys
sys.path.append("..")

import torch
import helper
import pathlib
import json
from pprint import pformat
import numpy as np
from sensorprocessing.sp_helper import load_picturefile_to_tensor


def create_RNN_training_sequence_xy(x_seq, y_seq, sequence_length):
    """Create supervised training data for RNNs such as LSTM from two sequences. In this data, from a string of length sequence_length in x_seq we are predicting the next item in y_seq. 
    Returns the results as tensors
    """
    # Prepare training data
    total_length = x_seq.shape[0]
    inputs = []
    targets = []
    for i in range(total_length - sequence_length):
        # Input is a subsequence of length `sequence_length`
        input_seq = x_seq[i:i + sequence_length]  
        # Shape: [sequence_length, latent_size]
        
        # Target is the next vector after the input sequence
        target = y_seq[i + sequence_length]       
        # Shape: [output_size]
        
        # Append to lists
        inputs.append(torch.tensor(input_seq))
        targets.append(torch.tensor(target))

    #inputs = np.array(inputs)
    #targets = np.array(targets)

    # Convert lists to tensors for tWraining
    inputs = torch.stack(inputs)   # Shape: [num_samples, sequence_length, latent_size]
    targets = torch.stack(targets) # Shape: [num_samples, latent_size]
    return inputs, targets


class BCDemonstration:
    """This class encapsulates loading a demonstration with the intention to convert it into training data.
    
    This code is a training helper which encapsulates one behavior cloning demonstration, which is a sequence of form $\{(s_0, a_0), ...(s_n, a_n)\}$. 

    In practice, however, we want to create a demonstration that maps the latent encodings to actions $\{(z_0, a_0), ...(z_n, a_n)\}$
    
    The transformation of $s \rightarrow z$ is done through an object of type  AbstractSensorProcessing. 

    In a practical way, the source of information for a BC demonstration is a demonstration directory, and the saved robot control there.
    """

    def __init__(self, source_dir, sensorprocessor, actiontype = "rc-position-target", camera = None):
        self.source_dir = source_dir
        self.sensorprocessor = sensorprocessor
        assert actiontype in ["rc-position-target", "rc-angle-target", "rc-pulse-target"]
        self.actiontype = actiontype
        # analyze the directory
        self.cameras, self.maxsteps = helper.analyze_demo(source_dir)
        # analyze 
        if camera is None:
            self.camera = self.cameras[0]
        else:
            self.camera = camera
        # read in _demonstration.json, load the trim values
        with open(pathlib.Path(self.source_dir, "_demonstration.json")) as file:
            data = json.load(file)
        self.trim_from = data["trim-from"]
        self.trim_to = data["trim-to"]
        if self.trim_to == -1:
            self.trim_to = self.maxsteps

    def read_z_a(self):
        """Reads in the demonstrations for z and a and returns them in the form of float32 numpy arrays"""
        z = []
        a = []
        for i in range(self.trim_from, self.trim_to):
            zval = self.get_z(i)
            # print(zval.cpu())
            z.append(zval)
            a.append(self.get_a(i))
        return np.array(z, dtype=np.float32), np.array(a, dtype=np.float32)

    def __str__(self):
        #return json.dumps(self.__dict__, indent=4)
        return pformat(self.__dict__)

    def get_z(self, i):
        filepath = pathlib.Path(self.source_dir, f"{i:05d}_{self.camera}.jpg")
        val = self.sensorprocessor.process_file(filepath)
        return val
    
    def get_image(self, i, transform = None):
        """Gets the image as a torch batch"""
        filepath = pathlib.Path(self.source_dir, f"{i:05d}_{self.camera}.jpg")
        sensor_readings, image = load_picturefile_to_tensor(filepath, transform)
        return sensor_readings, image
        
    def get_a(self, i):
        filepath = pathlib.Path(self.source_dir, f"{i:05d}.json") 
        with open(filepath) as file:
            data = json.load(file)
        if self.actiontype == "rc-position-target":
            datadict = data["rc-position-target"]
            a = list(datadict.values())
            return a
        if self.actiontype == "rc-angle-target":
            datadict = data["rc-angle-target"]
            a = list(datadict.values())
            return a
        if self.actiontype == "rc-pulse-target":
            datadict = data["rc-pulse-target"]
            a = list(datadict.values())
            return a