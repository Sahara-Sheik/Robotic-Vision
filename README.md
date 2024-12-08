# BerryPicker
This software package contains code for vision driven robot manipulation using an inexpensive robotic arm.

The type of skills we aim to develop include (but are not limited to) agricultural manipulations such as picking fruits and berries, checking the ripeness of fruits, or detecting plant diseases. 

### Obsolete from here

The recommended way to organize this code is as follows:

```
top directory
    \data
        \demos
            <<< this is where the demonstrations go
    \github
        \VisionBasedDataManipulator
            <<< this git checkout
    \venv
        .venv
            <<< this is where the Python 3.10 environment goes
```
### End obsolete

## Libraries needed
* approxeng.input == 2.5
* torch, torchvision, pandas, numpy
* matplotlib
* tqdm
* pyyaml
* tensorboardX
* pyserial
* opencv-python, opencv-contrib-python



Notes:
* The software for the gamepad input, approxeng.input needs to be at version 2.5, and it requires python not higher than 3.10 (as it needs something called evdev, which seem to have problems with higher python)

