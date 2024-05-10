import pathlib
import pickle
import tensorflow as tf

from CVAE_VisualModule import CVAE_VisualModule
from Datasets import Datasets
# replace this one with the local one when debugging or developing
# from RobotVisualController import RobotVisualController
import paths
import logging
#logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)


def get_trained_model(config):
    """Running the training of the representation module. This replaces the 
    one from from CVAE_VisualModule"""

    model_path = config["model_path"]
    epochs_target = config["epochs_target"]
    """Returns a visual encoder, either by loading the existing model, 
    or training the model on the images in the unsupervised dir"""
    model = CVAE_VisualModule(config)
    # loading or creating the config file that 
    config_path = pathlib.Path(model_path.parent, model_path.name + ".config")
    if config_path.exists():
        with open(config_path, "rb") as f:
            # FIXME: this overwrites the config...
            config = pickle.load(f)            
    else:
        # FIXME: it should be epoch_trained and start at 0
        #
        #config = {"epoch" : 1, "epochs_max": epochs_max}
        config["epochs_trained"] = 0
    if pathlib.Path(model_path.parent, model_path.name + ".index").exists():
        model.load_weights(str(model_path))
        logging.info(f"model loaded from {model_path}")
        if config["epochs_trained"] >= config["epochs_target"]:
            logging.info(f"Model fully trained to the required number of epochs {config['epochs_target']}")
            return model
    logging.debug(str(config))
    # model either does not exist of it is not fully trained 
    if config["load_only"]:
        raise Exception("was instructed to load only, this model needs training.")
    logging.info("Proceed to train the model")
    
    dataset = get_dataset(config)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    epoch_init = config["epochs_trained"]
    epochs_target = config["epochs_target"]
    for epoch in range(epoch_init, epochs_target+1):
        config["epochs_trained"] = epoch
        for batch in dataset:
            model.compute_apply_gradients(batch, optimizer)
        if epoch % 100 == 0:
            logging.info(f"training epoch {epoch} / {epochs_target}")
            with open(config_path, "wb") as f:
                pickle.dump(config, f)
            model.save_weights(str(model_path))
            # loss evaluation, on the first batch?                
    logging.info(f"training done, saving model to {model_path}")
    model.save_weights(str(config["model_path"]))
    return model


def get_dataset(config):
    """Deals with loading the correct dataset"""
    subdirs = [x for x in paths.demonstrations_path.iterdir() if x.is_dir()]
    print(subdirs)
    config["unsupervised_data.dirs"] = subdirs
    config["unsupervised_data.template"] = "camera_1_*.jpg"

    image_paths = []
    for dir in subdirs:
        for a in dir.iterdir():
            if a.match(config["unsupervised_data.template"]):
                image_paths.append(a)
    print(image_paths)
    config["unsupervised_data.imagepaths"] = image_paths
    config["unsupervised_data.do_resize"] = True
    dataset = Datasets.create_unsupervised_dataset_from_specified(config)
    return dataset

def main():


    print("Learn a representation")

    config = {
    "latent_dim": 50, "image_width": 32, "image_height": 32, "image_color_channels": 3,
    "epochs_target": 100, 
    "model_path": paths.visual_module_model_path, "load_only": False, "batch": 16
    }

    model = get_trained_model(config)

    print("Not implemented yet: specify the dataset (as a set of directories)") 
    print("Not implemented yet: train the representation component ") 
    print("Not implemented yet: save the trained neural network") 
    print("Not implemented yet: support for checkpoint and restart (time based")


    # model = CVAE_VisualModule.get_trained_model(config)




 

if __name__ == "__main__":
    main()