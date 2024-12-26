"""
This file contains the constants for the AL5D robot
"""

SERVO_Z = 0
SERVO_SHOULDER = 1
SERVO_ELBOW = 2
SERVO_WRIST = 3
SERVO_WRIST_ROTATION = 4
SERVO_GRIP = 5

# Minimums and maximums for the position values. 
# Most of the time the system should operate in ranges that are in the 
# middle. Exception is the gripper, which probably often open or closed



#POSITION_HEIGHT_DEFAULT = 5.0 # inches
#POSITION_HEIGHT_MIN = 1.0
#POSITION_HEIGHT_MAX = 10.0

#POSITION_DISTANCE_DEFAULT = 5.0 # inches
#POSITION_DISTANCE_MIN = 1.0
#POSITION_DISTANCE_MAX = 10.0

# rotation of the robot around - we assume -90 (left) to 90 right
# unclear if there is a real limit here
#POSITION_HEADING_DEFAULT = 0.0
#POSITION_HEADING_MIN = -90.0
#POSITION_HEADING_MAX = 90.0

# wrist angle
#POSITION_WRIST_ANGLE_DEFAULT = -45.0
#POSITION_WRIST_ANGLE_MIN = -90.0 # pointing down
#POSITION_WRIST_ANGLE_MAX = 90.0 # point up

# wrist rotation
#POSITION_WRIST_ROTATION_DEFAULT = 75.0 # the straight value????
#POSITION_WRIST_ROTATION_MIN = POSITION_WRIST_ROTATION_DEFAULT - 90.0 # FIXME
#POSITION_WRIST_ROTATION_MAX = POSITION_WRIST_ROTATION_DEFAULT + 90.0

# gripper: between 0 and 100
#POSITION_GRIPPER_DEFAULT = 100.0
#POSITION_GRIPPER_MAX = 100.0
#POSITION_GRIPPER_MIN = 0.0


# Constants - Speed in µs/s, 4000 is roughly equal to 360°/s or 60 RPM
#           - A lower speed will most likely be more useful in real use, such as 100 µs/s (~9°/s)
CST_ANGLE_MIN = 0
CST_ANGLE_MAX = 180
CST_PULSE_MIN = 500
CST_PULSE_MAX = 2500

# fixme this might be too large a range, but it is good for debugging
#CST_PULSE_MIN = 0
#CST_PULSE_MAX = 3000

CST_SPEED_MAX = 4000
CST_SPEED_FAST = 500
CST_SPEED_DEFAULT = 100
# time: in microseconds to travel from the current position to the
# target position FIXME: this doesn't look like microseconds it is more like milliseconds
TIME_DEFAULT = 50

# the specific angle limits that need to be set for the servo by servo basis
# this is probably specific for the AL5D as a robot design. Each is a value
# [minimum, rest, maximum]
ANGLE_LIMITS = {SERVO_Z: [0, 90, 180],
                SERVO_SHOULDER: [0, 90, 180],
                SERVO_ELBOW: [0, 90, 180],
                SERVO_WRIST: [0, 90, 180],
                SERVO_WRIST_ROTATION: [0, 90, 180],
                SERVO_GRIP: [0, 90, 180]
                }

# the specific pulse correction that need to be added when translating
# the angle to pulse.
PULSE_CORRECTION = {SERVO_Z: 40, # there is some correction needed here...
                    SERVO_SHOULDER: -40,
                    SERVO_ELBOW: 0,
                    SERVO_WRIST: 0,
                    SERVO_WRIST_ROTATION: 0,
                    SERVO_GRIP: 0
                    }
