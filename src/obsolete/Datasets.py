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
import time

class Datasets:
                
    @staticmethod
    def create_unsupervised_dataset(image_dir, batch = 16, img_height = 32, img_width = 32, do_resize = False):
        """Create a dataset for training an autoencoder (unsupervised data) 
        from all the image files in the top level of the image_dir directory.
        The image files are loaded first in the memory, resized and put in an 
        array. Then we create the dataset from those tensor slices.
        FIXME: this might be somewhat expensive, and one needs to keep it in 
        the memory
        """
        logging.info(f"Started creating unsupervised dataset from {image_dir}")
        pic_list = []
        count = 0
        for image_filename in pathlib.Path(image_dir).iterdir():
            # print(image_filename)
            img = tf.io.read_file(str(image_filename))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            if do_resize:
                img = tf.image.resize(img, [img_width, img_height])
            pic_list.append(img)
            count = count + 1
            if count % 5000 == 0:
                logging.info(f"At image: {count}")

        logging.info("About to create dataset from tensor slices")
        dataset = tf.data.Dataset.from_tensor_slices(pic_list)
        logging.info("About to shuffle")
        dataset = dataset.shuffle(1000) # buffer size 1000
        logging.info("About to batch")
        dataset = dataset.batch(batch, drop_remainder = True)
        logging.info(f"Done creating unsupervised dataset from {image_dir} with {count} images")
        return dataset

    @staticmethod
    def create_unsupervised_dataset_from_specified(config):
        "Creates a dataset for training an autoencoder. The images are supposed to be in all the directories specified in the config"
        logging.info(f"Started creating unsupervised dataset from the specified directories")
        pic_list = []
        count = 0
        for image_filename in config["unsupervised_data.imagepaths"]:
            img = tf.io.read_file(str(image_filename))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            if config["unsupervised_data.do_resize"]:
                img = tf.image.resize(img, [config["image_width"], config["image_height"]])
            pic_list.append(img)
            count = count + 1
            if count % 5000 == 0:
                logging.info(f"At image: {count}")
        logging.info("About to create dataset from tensor slices")
        dataset = tf.data.Dataset.from_tensor_slices(pic_list)
        logging.info("About to shuffle")
        dataset = dataset.shuffle(1000) # buffer size 1000
        logging.info("About to batch")
        dataset = dataset.batch(config["batch"], drop_remainder = True)
        logging.info(f"Done creating unsupervised dataset from the specified directories with {count} images")
        return dataset



    @staticmethod
    def create_image_control_demonstration_dataset(control_path, pictures_dir, maxcnt=999):
        """Creates a demonstration dataset of (image, control) pairs. 
        The input is the format chosen by Rouhollah. For the given task, at the top of the directory there is a
        text cvs file with the columns:

        time,task,user,robot,reward,gripper,joint1,joint2,joint3,joint4,joint5,joint6

        There is a picture directory where there is a collection of jpg pictures 0.jpg... 

        There is one-to-one mapping between the lines in the text file and the pictures. 
        The dataset generated is created as 2 columns "image" and "control". 
        The control is only the gripper and joint values. 
        """
        # Load the file and read it line by line
        logging.info(f"Started creating image control demonstration dataset from {control_path}")
        f = control_path.open()
        controls = []
        images = []
        for line, text in enumerate(f):
            if line < 3:
                continue
            count = line - 3
            if count == maxcnt:
                break
            control = np.fromstring(text, sep=",")
            # get the gripper value and the 6 joint values
            control = control[4:11]
            # print(control)
            image_filename = pathlib.Path(pictures_dir, f"{count}.jpg")
            if image_filename.exists():
                img = tf.io.read_file(str(image_filename))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.convert_image_dtype(img, tf.float32)
                count = count + 1
            else:
                raise Exception(f"Could not find image {image_filename}")
                # logging.warning(f"Could not find image {image_filename} - skipping")
                continue
            # create the pairs
            controls.append(control)
            images.append(img)
            if line % 5000 == 0:
                logging.info(f"At line: {line}")
        df_tensor = {"image" : images, "control": controls}
        dataset = tf.data.Dataset.from_tensor_slices(df_tensor)
        logging.info(f"Done creating image control demonstration dataset, total size {len(controls)}")
        return dataset