"""
This file contains settings for the VisionBasedRobotManipulator. These can be considered as constants for a particular run, but they might have different values on different computers or setups.

Should access them through settings.NAME
"""

# The USB port where the AL5D robot is connected on this particular computer.
# alternatives: linux: /dev/ttyUSB0, /dev/ttyUSB1
ROBOT_USB_PORT = "/dev/ttyUSB0" 

# The camera numbers which should be used for the robot, as a list
# Typically 0 is the webcam of the computer, other ones in the order they are connected
ACTIVE_CAMERA_LIST = [2] 


