import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import logging
from Datasets import Datasets


class CVAE_VisualModule(tf.keras.Model):
    """ A convolutional variational autoencoder """
    
    def __init__(self, config):
        """ Create the CVAE model for a specific image size, and size of latent dimensions"""
        super().__init__()
        
        latent_dim = config["latent_dim"]
        size_x = config["image_width"]
        size_y = config["image_height"]
        color_channels = config["image_color_channels"]
        # this is how small it gets through successive decompositions
        small_x = int(size_x / 4)
        small_y = int(size_y / 4)
        
        # the inference network
        self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(size_x, size_y, color_channels)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
        )

        # the generative net
        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=small_x * small_y * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(small_x, small_y, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        """ Sampling a number of samples (by default 100 random generated ones) from the encoding, and decode them."""
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode_with_variance(self, x):
        """ 
        Takes a batch of images x. Returns a batch of mean encodings, and a batch of log variance encodings.
        """
        encoding = self.inference_net(x)
        mean, logvar = tf.split(encoding, num_or_size_splits=2, axis=1)
        return mean, logvar

    def encode(self, x):
        """ 
        Takes a batch of images x and returns a batch of encodings
        """
        return self.encode_with_variance(x)[0]
    
    
    def reparameterize(self, mean, logvar):
        """Performs the re-parametrization of the network by generating random values following the computed 
        log-variance and mean - the output of this one is a z value. 
        What re-parametrization means in this case is that this is going to be a random number. But we are pushing
        in here a uniform distribution, so the parameters of this value will be the parameters that we used to 
        calculated the mean and logvar - that is the parameters of the inference network. 
        
        FIXME: I don't understand what the 0.5 mean here?
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """Decodes a certain z value by running the generative net, and possibly applying a sigmoid. 
        We might not apply the sigmoid, because we have loss expressions that directly take a logit. 
        We might apply the sigmoid, because it undoes the logit. 
        Then we simply interpret it not as a probability value, but as a color channel value. """
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @staticmethod   
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        """Implements the calculation of the log-normal probability density function.
        Lotzi: if I understand correctly, this returns the sum of the probability for a certain sample, 
        when given the mean and the log variance for the probability. It does this for a vector."""
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @tf.function
    def compute_loss(self, x):
        """ Calculates the loss of the VAE for an input batch x"""
        # Runs through the model, and gets the output as a logit
        mean, logvar = self.encode_with_variance(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid = False)    
        # calculates the cross entropy autoencoding loss, and then sums it up along x, y and color (is it???)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # the other component of the VAE loss is the KL divergence, 
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        L = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        return L

    @tf.function
    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)            
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # this won't work like this: it is only executed once at tracing
            # print(float(loss))
            tf.print(loss)
    
    @staticmethod 
    def get_trained_model(config):
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
        dataset = Datasets.create_unsupervised_dataset(config["training_data_dir"])
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