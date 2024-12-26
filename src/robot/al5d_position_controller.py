from . import al5d_constants
import numpy as np
# import serial
# import time
from .al5d_helper import RobotHelper
from .al5d_pulse_controller import PulseController
from .al5d_angle_controller import AngleController
from math import sqrt, atan, acos, fabs, degrees
from copy import copy
import logging

logging.basicConfig(level=logging.WARNING)

POS_DEFAULT = {"height": 5.0, "distance": 5.0, "heading": 0.0, 
               "wrist_angle": -45.0, "wrist_rotation": 75.0, "gripper": 100}

#POS_MIN = {"height": 1.0, "distance": 1.0, "heading": -90.0, 
#               "wrist_angle": -90.0, "wrist_rotation": 75.0 - 90.0, 
#               "gripper": 0}

#POS_MAX = {"height": 5.0, "distance": 10.0, "heading": 90.0, 
#               "wrist_angle": 90.0, "wrist_rotation": 75.0 + 90.0, 
#               "gripper": 100}

# Handwired wrist-rotation for a much shorter range as it was creating problems.

POS_MIN = {"height": 1.0, "distance": 3.0, "heading": -90.0, 
               "wrist_angle": -90.0, "wrist_rotation": 60.0, 
               "gripper": 0}

POS_MAX = {"height": 5.0, "distance": 10.0, "heading": 90.0, 
               "wrist_angle": 0.0, "wrist_rotation": 90.0, 
               "gripper": 100}


class RobotPosition:
    """A data class describing the high level robot position"""

    def __init__(self, values = None):
        if values is None:
            self.values = copy(POS_DEFAULT)
        else:
            self.values = copy(values)
    
    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    @staticmethod
    def limit(posc):
        """Verifies whether the given position is safe, defined between the mind and the max"""
        retval = True
        for fld in posc.values:
            retval = retval and posc.values[fld] <= POS_MAX[fld]
            retval = retval and posc.values[fld] >= POS_MIN[fld]
        return retval

    def to_normalized_vector(self):
        """Converts the positions to a normalized vector"""
        retval = np.zeros(6, dtype = np.float32)
        for i, fld in enumerate(self.values):
            retval[i] = RobotHelper.map_ranges(self.values[fld], POS_MIN[fld], POS_MAX[fld])
        return retval

    @staticmethod
    def from_normalized_vector(values):
        """Creates the rp from a normalized numpy vector"""
        rp = RobotPosition()
        for i, fld in enumerate(rp.values):
            rp.values[fld] = RobotHelper.map_ranges(values[i], 0.0, 1.0, POS_MIN[fld], POS_MAX[fld])
        return rp

    @staticmethod
    def from_vector(values):
        """Creates a RobotPosition from a numpy vector"""
        rp = RobotPosition()
        for i, fld in enumerate(rp.values):
            rp.values[fld] = values[i]
        return rp

    def empirical_distance(self, other):
        """A weighted distance function between two robot positions"""
        w = np.ones([6]) / 6.0
        norm1 = np.array(self.to_normalized_vector())
        norm2 = np.array(other.to_normalized_vector())
        val = np.inner(w, np.abs(norm1 - norm2))
        return val    

    def __str__(self):
        retval = "Position: "
        for fld in self.values:
            v = self.values[fld]            
            retval += f" {fld}:{v:.2f}"
        return retval

class PositionController:
    """A controller that controls the robot in terms of the physical position of the actuator. The general idea is that this captures some of the low level calculations necessary to control the robot in an intelligent way. The idea is that this had been engineered, while what comes on top of this will be learned."""

    def __init__(self, device = '/dev/ttyUSB0'):
        self.pulse_controller = PulseController(device = device)
        self.pulse_controller.start_robot()
        self.angle_controller = AngleController(self.pulse_controller)
        self.pos = RobotPosition()
        self.move(self.pos)

    def get_position(self):
        return copy(self.pos)

    def stop_robot(self):
        self.pulse_controller.stop_robot()

    @staticmethod
    def ik_shoulder_elbow_wrist_old(target:RobotPosition):
        """Performs the inverse kinematics necessary to the height and distance"""
        # if AL5D - a set of constants that are used in the
        A = 5.75
        B = 7.375
        #rtod = 57.295779  # Radians to degrees constant

        # position_distance should be larger than zero
        if target.distance <= 0:
            raise Exception("x <= 0")

        #angle_elbow = 0
        #angle_shoulder = 0
        #angle_wrist = 0
        # Get distance and check it for error
        m = sqrt((target.height * target.height) + (target.distance * target.distance))
        # this cannot happen, I think
        #if(m <= 0):
        #    raise Exception("m <= 0")
        # Get first angle (radians)
        a1 = degrees( atan(target.height / target.distance) )
        # Get 2nd angle (radians)
        a2 = degrees( acos((A * A - B * B + m * m) / ((A * 2) * m)) )
        #	print("floatA2       = " + str(floatA2))

        # Calculate elbow angle (radians)
        angle_elbow =  degrees( acos((A * A + B * B - m * m) / ((A * 2) * B)) )
        #	print("floatElbow    = " + str(floatElbow))

        # Calculate shoulder angle (radians)
        angle_shoulder = a1 + a2
        #	print("floatShoulder = " + str(floatShoulder))

        # Obtain angles for shoulder / elbow
        #angle_elbow = floatElbow * rtod
        #	print("Elbow         = " + str(floatA2))
        #angle_shoulder = floatShoulder * rtod
        #	print("Shoulder      = " + str(Shoulder))

        # Check elbow/shoulder angle for error
        if (angle_elbow <= 0) or (angle_shoulder <= 0):
            raise Exception("Elbow <=0 or Shoulder <=0")
        angle_wrist = fabs(target.wrist_angle - angle_elbow - angle_shoulder) - 90

        # corrections compared to the system I got
        angle_elbow = 180 - int(angle_elbow) - 20         
        angle_shoulder = int(angle_shoulder)
        # It seems that this goes in the opposite direction - or the way they added it up in the calculation was incorrect and you need the elbow removed
        angle_wrist = 180 - int(angle_wrist) + 25 # zero is vertical
        return angle_shoulder, angle_elbow, angle_wrist

    @staticmethod
    def ik_shoulder_elbow_wrist(target:RobotPosition):
        """Performs the inverse kinematics necessary to the height and distance"""
        # if AL5D - a set of constants that are used in the
        A = 5.75
        B = 7.375
        # position_distance should be larger than zero
        if target["distance"] <= 0:
            raise Exception("x <= 0")
        # Get distance and check it for error
        m = sqrt((target["height"] * target["height"]) + (target["distance"] * target["distance"]))
        a1 = degrees( atan(target["height"] / target["distance"]) )
        # Get 2nd angle (radians)
        a2 = degrees( acos((A * A - B * B + m * m) / ((A * 2) * m)) )
        # Calculate elbow angle (radians)
        angle_elbow =  degrees( acos((A * A + B * B - m * m) / ((A * 2) * B)) )
        # Calculate shoulder angle (radians)
        angle_shoulder = a1 + a2
        # Check elbow/shoulder angle for error
        if (angle_elbow <= 0) or (angle_shoulder <= 0):
            raise Exception("Elbow <=0 or Shoulder <=0")
        angle_wrist = fabs(target["wrist_angle"] - angle_elbow - angle_shoulder) - 90
        # corrections compared to the system I got
        angle_elbow = 180 - int(angle_elbow) - 20         
        angle_shoulder = int(angle_shoulder)
        # It seems that this goes in the opposite direction - or the way they added it up in the calculation was incorrect and you need the elbow removed
        angle_wrist = 180 - int(angle_wrist) + 25 # zero is vertical
        return angle_shoulder, angle_elbow, angle_wrist


    @staticmethod
    def ik_shoulder_elbow_wrist_old(target:RobotPosition):
        """Performs the inverse kinematics necessary to the height and distance"""
        # if AL5D - a set of constants that are used in the
        A = 5.75
        B = 7.375
        # position_distance should be larger than zero
        if target.distance <= 0:
            raise Exception("x <= 0")

        #angle_elbow = 0
        #angle_shoulder = 0
        #angle_wrist = 0
        # Get distance and check it for error
        m = sqrt((target.height * target.height) + (target.distance * target.distance))
        # this cannot happen, I think
        #if(m <= 0):
        #    raise Exception("m <= 0")
        # Get first angle (radians)
        a1 = degrees( atan(target.height / target.distance) )
        # Get 2nd angle (radians)
        a2 = degrees( acos((A * A - B * B + m * m) / ((A * 2) * m)) )
        #	print("floatA2       = " + str(floatA2))

        # Calculate elbow angle (radians)
        angle_elbow =  degrees( acos((A * A + B * B - m * m) / ((A * 2) * B)) )
        #	print("floatElbow    = " + str(floatElbow))

        # Calculate shoulder angle (radians)
        angle_shoulder = a1 + a2
        #	print("floatShoulder = " + str(floatShoulder))

        # Obtain angles for shoulder / elbow
        #angle_elbow = floatElbow * rtod
        #	print("Elbow         = " + str(floatA2))
        #angle_shoulder = floatShoulder * rtod
        #	print("Shoulder      = " + str(Shoulder))

        # Check elbow/shoulder angle for error
        if (angle_elbow <= 0) or (angle_shoulder <= 0):
            raise Exception("Elbow <=0 or Shoulder <=0")
        angle_wrist = fabs(target.wrist_angle - angle_elbow - angle_shoulder) - 90

        # corrections compared to the system I got
        angle_elbow = 180 - int(angle_elbow) - 20         
        angle_shoulder = int(angle_shoulder)
        # It seems that this goes in the opposite direction - or the way they added it up in the calculation was incorrect and you need the elbow removed
        angle_wrist = 180 - int(angle_wrist) + 25 # zero is vertical
        return angle_shoulder, angle_elbow, angle_wrist



    def move(self, target: RobotPosition):
        """Move to the specified target position: new version with one shot commands"""
        angle_z = 90 + target["heading"]
        angle_shoulder, angle_elbow, angle_wrist = self.ik_shoulder_elbow_wrist(target)
        angle_wrist_rotation = target["wrist_rotation"]
        # safety check here
        angles = np.zeros(5)
        angles[al5d_constants.SERVO_ELBOW] = angle_elbow
        angles[al5d_constants.SERVO_SHOULDER] = angle_shoulder
        angles[al5d_constants.SERVO_WRIST] = angle_wrist
        angles[al5d_constants.SERVO_WRIST_ROTATION] = angle_wrist_rotation
        angles[al5d_constants.SERVO_Z] = angle_z
        self.angle_controller.control_angles(angles, target["gripper"])
        self.pos = target

