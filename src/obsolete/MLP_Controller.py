import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython import display
import pickle
import logging
import platform
# the finished version of this code
from CVAE_VisualModule import CVAE_VisualModule
from Datasets import Datasets
import paths

class MLP_Controller(tf.keras.Model):
    """A simple implementation of a robot controller, as a two layer regression network"""
    
    def __init__(self, config):
        """ Create a controller with the specified latent dimensions and control dimensions
        latent_dim: the dimensionality of the input, the size of the z 
        control_dim: the dimensionality of the output, the size of the robot controls
        """
        super().__init__()
        
        self.latent_dim = config["latent_dim"]
        self.control_dim = config["control_dim"]
        self.network = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim, )),
            tf.keras.layers.Dense(30),
            tf.keras.layers.Dense(self.control_dim),
          ]
        )
        self.network.summary()
        

    def predict(self, z):
        """Make a prediction"""
        actions = self.network.predict(z)
        return actions
    
    @staticmethod
    def get_trained_model(config, visual_module=None):
        """Returns a controller, either by loading the already trained model, or training a model on the
        demonstrations in the demonstration_dir"""
        
        model_path = config["model_path"]
        demonstration_control_path = config["demonstration_control_path"]
        demonstration_images_dir = config["demonstration_images_dir"]
        latent_dim = config["latent_dim"]
        control_dim = config["control_dim"]
        load_only = config["load_only"]
        
        model = MLP_Controller(config)
        model.network.compile(loss=
                           'mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        config_path = pathlib.Path(model_path.parent, model_path.name + ".config")
        if config_path.exists():
            with open(config_path, "rb") as f:
                # FIXME: 
                config = pickle.load(f)            
        else:            
            config["epochs_trained"] = 0
        if pathlib.Path(model_path.parent, model_path.name + ".index").exists():
            model.load_weights(str(model_path))
            logging.info(f"model loaded from {model_path}")
            if config["epochs_trained"] >= config["epochs_target"]:
                logging.info(f"Model fully trained to the required number of epochs {config['epochs_target']}")
                return model
        if config["load_only"]:
            raise Exception(f"Was instructed to load only, this model needs training {model_path}")
        logging.info("Proceed to train the model")
        # create the demonstration data
        ds = Datasets.create_image_control_demonstration_dataset(demonstration_control_path, demonstration_images_dir)
        x = []
        y = []
        #for a in ds.take(10):
        for a in ds:
            y.append(a["control"])
            imgbatch = np.array([a["image"]])            
            # we treat this as the mean, and unpack it from the array
            h = visual_module.encode(imgbatch)
            x.append(h[0])    
            #print(a)
        x = np.array(x)
        y = np.array(y)
        # 
        epoch_init = config["epochs_trained"]
        epochs_target = config["epochs_target"]
        for epoch in range(epoch_init, epochs_target+1):
            ### validation_split???
            hist = model.network.fit(x, y, verbose=0, validation_split=0.2, epochs=1)
            # FIXME: Does this miscount it???
            config["epochs_trained"] = epoch+1
            #hist.history.keys()
            if epoch % 100 == 0:
                logging.info(f"Loss: {hist.history['loss'][-1]} validation loss {hist.history['val_loss'][-1]}")
                logging.info(f"training epoch {epoch} / {epochs_target}")
                
                with open(config_path, "wb") as f:
                    pickle.dump(config, f)                    
                model.save_weights(str(model_path))
        return model        
        