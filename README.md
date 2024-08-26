# VisionBasedRobotManipulator
Collecting demonstrations and perform policy learning for an inexpensive, vision driven robot.

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

Notes:
* The software for the gamepad input, approxeng.input needs to be at version 2.5, and it requires python not higher than 3.10 (as it needs something called evdev, which seem to have problems with higher python)