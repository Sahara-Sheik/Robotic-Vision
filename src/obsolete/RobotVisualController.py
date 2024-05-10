import numpy as np
from CVAE_VisualModule import CVAE_VisualModule
from MLP_Controller import MLP_Controller

class RobotVisualController:
    """This class contains the implementation of the robot controller. It has various networks which are
    trained in various setups."""
    
    @staticmethod
    def generate_visual_module(config):
        """Factory method to generate a visual module according to the configuration. 
        While it would be somewhat convenient to pass the config file over, I am going to recreate here 
        the config file, to make the transferred data more visible"""
        name = config["visual_module"]
        if name == "CVAE":
            cvaeconfig = {"latent_dim": config["h_size"], "image_width": config["image_width"], 
                          "image_height": config["image_height"], 
                          "image_color_channels": config["image_color_channels"], 
                          "training_data_dir": config["visual_module_training_data_dir"],
                          "model_path": config["visual_module_model_path"],
                          "epochs_target": config["visual_module_epochs_target"],
                          "load_only": False 
                         }
            # FIXME: the CVAE visual module does not take this in this way
            # FIXME: separate component when I want load only vs training
            # 
            module = CVAE_VisualModule.get_trained_model(cvaeconfig)
            return module
        if name == "CVAE-GAN":
            raise Exception("CVAE-GAN visual module not implemented yet")
        if name == "VGG":
            raise Exception("VGG visual module not implemented yet")
        
    @staticmethod
    def generate_control_module(config, visual_module):
        """Factory method to generate a control module according to the configuration"""
        name = config["control_module"]
        if name == "MLPController":
            controllerconfig = {
                "latent_dim": config["h_size"], "control_dim": config["a_size"], 
                "epochs_target": config["control_module_epochs_target"], 
                "demonstration_dir": config["control_module_training_data_dir"], 
                "model_path": config["control_module_model_path"], 
                "epochs_target": config["control_module_epochs_target"],
                "load_only": False }
            controller = MLP_Controller.get_trained_model(controllerconfig, visual_module)
            return controller
        if name == "LSTMController":
            raise Exception("LSTMController not implemented yet")
    
    def __init__(self, config):
        """Creates a robotic initialization controller, and tries to initialize it, if possible"""
        self.config = config
        self.visual_module = self.generate_visual_module(config)
        self.control_module = self.generate_control_module(config, self.visual_module)        
            
    def control(self, image):  
        imagebatch = np.array([image])
        h = self.visual_module.encode(imagebatch)
        a = self.control_module.predict(h)
        return a