# Learning a representation. For the time being, this code is learning a convolutional VAE.

import settings
from sensorprocessing.conv_vae import get_conv_vae_config, train


def main():
    print("Learn a representation (the vision component)")
    config = get_conv_vae_config()
    train(config)

if __name__ == "__main__":
    main()

