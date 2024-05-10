import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense




class LSTM_Controller(tf.keras.Model):
    """An LSTM based implementation of the robot controller. It takes the input as a time series"""
    
    def __init__(self, config):
        """Create a controller, with the specified latent dimensions, control dimensions
        FIXME: will also do the number of LSTM layers"""
        super().__init__()                        
        # units - dimensionality of the output space, I assume also the internal state
        # should this be something else??
        # self.inputlayer = Input(shape=(self.latent_dim,))
        self.lstm_1 = LSTM(units=config["control_dim"], input_shape=(config["timesteps"], config["control_dim"]), return_state=True)
        
    def call(self, inputs):
        #val1 = self.inputlayer(inputs)
        #outputs, state = self.lstm_1(inputs)
        outputs, state_h, state_c = self.lstm_1(inputs)
        return outputs


def generate_timeseries(dim = 7, length = 100):
    """Generates timeseries of the specified length. 
    Each step is a set of features of dimensionality dim 
    The steps evolve through a set of summed epicycles with different speed and initial angle"""
    scale = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05, 0.02] # the "length" of the arm
    omega = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.1] # angular velocity radian / second
    phase = [0.5, 0.5, -0.5, 0.5, 1, 1.25, -1.0] # initial phase radian
    retval = []
    for t in range(length):
        val = []
        acc = 0
        for j in range(dim):
            acc += scale[j] * math.sin(phase[j] + omega[j]*t)
            val.append(acc)
        retval.append(val)
    return np.array(retval)

def visualize_timeseries(val, x = 6, y = 0, label="unknown"):
    # visualizes a timeseries by plotting the 6 and 0 as x and y
    #print(val[:,0])
    ax = plt.subplot()
    ax.plot(val[:,x], val[:,y], label = label)
    ax.legend()
    #plt.show()

def generate_training_data(inputdata, config, skip = 7):
    # generate the training data from an input stream, by sampling every 7 th item. The x part is a string of inputs, the y part is the item after that.

    x_train = []
    y_train = []
    for i in range( int((len(inputdata) - config["timesteps"]) / skip)):
        base = i * skip
        x = inputdata[base:base+config["timesteps"]] # input to a point
        y = inputdata[base + config["timesteps"]] # output after that
        x_train.append(x)
        y_train.append(y)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def create_and_train_model(x_train, y_train, config):
    model = LSTM_Controller(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mse")
    model.build(input_shape=(config["batch"], config["timesteps"], config["control_dim"]))
    model.summary()    
    model.fit(x_train, y_train, epochs = config["epochs"])
    return model

# inputbatch = np.array([ val[50],])
### Weird, it needs to explicitly set to float32!!!
#inputbatch = np.array([[ [inputdata[50]] ,[inputdata[51]] , [inputdata[52]],[inputdata[53]], [inputdata[54]]]], dtype="float32")

# generate some database, and create the 

config = {"latent_dim": 1, "control_dim": 7, "learning_rate": 0.01, "batch": 1, "timesteps": 5, "features": 1, "epochs": 20}
# was epochs 200

val = generate_timeseries(dim=config["control_dim"], length = 2000)
# inputdata = np.array([[i] for i in range(1000)], dtype="float32")
# print(val)
# visualize_timeseries(val[0:200])

x_train, y_train = generate_training_data(val[0:1000], config, skip=1)
model = create_and_train_model(x_train, y_train, config)

## trying it out
## in this example, we are generating it based on the training data at each 
## step, what if we make it recurse?

def predict_on_sequence(config, model, val, start, length):
    """ Use the model to make predictions by assuming that LSTM will always receive a correct value """
    x_test, y_test = generate_training_data(val[start:start+length+config["timesteps"]], config, skip=1)
    outputval = []
    for i in range(length):
        input = np.array([x_test[i]])
        output = model(input)
        #print(output)
        outputval.append(output[0])
    outputval = np.array(outputval)
    print(outputval.shape)
    return outputval

## in this example, we start from a point, and then use the output to generate them.
def predict_recursive(config, model, val, start, length):
    """ Use the model to make predictions by assuming that prediction of the LSTM is going to be recirculated into the input"""
    x_test, y_test = generate_training_data(val[start:start+length], config, skip=1)
    outputval2 = []
    input = np.array([x_test[0]])
    print(f"Input shape {input.shape}")
    for i in range(length):
        output = model(input)
        #print(output)
        outputval2.append(output[0])
        # create new input
        input2 = np.concatenate((input[0][1:], np.array([output[0]])))
        # print(f"Input 2 shape {input2.shape}")
        input = np.expand_dims(input2, axis=0)
    outputval2 = np.array(outputval2)
    print(outputval2.shape)
    return outputval2

#visualize_timeseries(val[1000:1800], 0, 1, "original" )
#visualize_timeseries(outputval[0:800], 0, 1, "predicted")

start = 300
delta = 10 # was 800

outputval_pred = predict_on_sequence(config, model, val, start, delta)
outputval_recursive = predict_recursive(config, model, val, start, delta)

visualize_timeseries(val[start + config["timesteps"]:start+ config["timesteps"] + delta], 0, 1, "original" )
visualize_timeseries(outputval_pred[0:delta], 0, 1, "predicted")
visualize_timeseries(outputval_recursive[0:delta], 0, 1, "predicted-recursive")

plt.show()

#inputbatch = np.expand_dims(inputdata[50:55], 0)
#print(inputbatch)
#retval = model(inputbatch)
#print(retval)
