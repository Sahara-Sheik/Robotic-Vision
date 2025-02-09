# What is this

The ```sensorprocessing''' package contains code to process the robots' sensor input and output an __encoding vector z__. The sensor processing code is usually some learned encoding model. As of Feb 2025, this is a single camera vision input. The use of other type of sensory information is planned for the future. 

The size of the encoding vector is specified in the __experiments__ association with these models. The experiments are named sensorprocessing_Foo, and they are in the experiment_configs folder. The experiments also specify the data sets used to train the encoding. 

Train_Foo notebooks contain code to train the model Foo.

Verify_Foo notebooks contain code to verify the learned model Foo. This can be done visually or numerically.

## Models (as of Feb 2025)

* ConvVAE: a convolutional variational autoencoder. 
* ProprioTunedVGG19: a VGG19 model tuned and dimensionality reduced on proprioception training data. 




