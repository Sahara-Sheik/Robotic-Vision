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


class RobotPosition:
    """A data class describing the high level robot position"""

    def __init__(self):
        # the height of the wrist from the table, in inches
        # FIXME: this should actually be the height of the gripper 
        self.height = al5d_constants.POSITION_HEIGHT_DEFAULT
        # the distance of the wrist from the center, in inches
        # FIXME: this should actually be the distance of the tip of the gripper
        self.distance = 5.0
        # the angle of the robot around the z axis
        self.heading = al5d_constants.POSITION_HEADING_DEFAULT
        # the angle formed by the wrist with the vertical 0 horizontal, -90 pointing down        
        self.wrist_angle = al5d_constants.POSITION_WRIST_ANGLE_DEFAULT
        # the position of the wrist rotator: 0 perpendicular to the arm direction, -90 left, +90 right
        self.wrist_rotation = 75.0 # the straight value?
        # the gripper position: 0 fully closed, 100 fully open
        self.gripper = al5d_constants.POSITION_GRIPPER_DEFAULT
    
    @staticmethod
    def limit(posc):
        """Verifies whether the proposed position is safe. If true, returns the True and same position. If false, returns False and a limited position.
        FIXME: for the time being, this is just a limit check, but it can get more complex, due to interactions between the stuff. 
        """
        pos = copy(posc)
        ok = True

        pos.height, check = RobotHelper.constrain(pos.height, al5d_constants.POSITION_HEIGHT_MIN, al5d_constants.POSITION_HEIGHT_MAX)
        ok = ok and not check

        pos.distance, check = RobotHelper.constrain(pos.distance, al5d_constants.POSITION_DISTANCE_MIN, al5d_constants.POSITION_DISTANCE_MAX)
        ok = ok and not check

        pos.heading, check = RobotHelper.constrain(pos.heading, al5d_constants.POSITION_HEADING_MIN, al5d_constants.POSITION_HEADING_MAX)
        ok = ok and not check

        pos.wrist_angle, check = RobotHelper.constrain(pos.wrist_angle, al5d_constants.POSITION_WRIST_ANGLE_MIN, al5d_constants.POSITION_WRIST_ANGLE_MAX)
        ok = ok and not check

        pos.wrist_rotation, check = RobotHelper.constrain(pos.wrist_rotation, al5d_constants.POSITION_WRIST_ROTATION_MIN, al5d_constants.POSITION_WRIST_ROTATION_MAX)
        ok = ok and not check

        pos.gripper, check = RobotHelper.constrain(pos.gripper, al5d_constants.POSITION_GRIPPER_MIN, al5d_constants.POSITION_GRIPPER_MAX)
        ok = ok and not check

        return ok, pos

    def as_dict(self):
        """Returns the position as a dictionary, for saving purposes"""
        retval = {}
        retval["height"] = self.height
        retval["distance"] = self.distance
        retval["heading"] = self.heading
        retval["wrist_angle"] = self.wrist_angle
        retval["wrist_rotation"] = self.wrist_rotation
        retval["gripper"] = self.gripper
        return retval

    @staticmethod
    def from_dict(value):
        """Creates the robot position from a dictionary """
        retval = RobotPosition()
        retval.height = value["height"]
        retval.distance = value["distance"]
        retval.heading = value["heading"]
        retval.wrist_angle = value["wrist_angle"]
        retval.wrist_rotation = value["wrist_rotation"]
        retval.gripper = value["gripper"]
        return retval 

    def to_normalized_vector(self):
        """Converts the positions to a normalized vector"""
        retval = np.zeros(6, dtype = np.float32)
        # 0 height
        retval[0] = RobotHelper.map_ranges(self.height, 
                                           al5d_constants.POSITION_HEIGHT_MIN, al5d_constants.POSITION_HEIGHT_MAX)
        # 1 distance
        retval[1] = RobotHelper.map_ranges(self.distance, 
                                           al5d_constants.POSITION_DISTANCE_MIN, 
                                           al5d_constants.POSITION_DISTANCE_MAX)
        # 2 heading
        retval[2] = RobotHelper.map_ranges(self.heading, 
                                           al5d_constants.POSITION_HEADING_MIN, 
                                           al5d_constants.POSITION_HEADING_MAX)
        # 3 wrist_angle
        retval[3] = RobotHelper.map_ranges(self.wrist_angle, 
                                           al5d_constants.POSITION_WRIST_ANGLE_MIN, 
                                           al5d_constants.POSITION_WRIST_ANGLE_MAX)
        # 4 wrist_rotation
        retval[4] = RobotHelper.map_ranges(self.wrist_rotation, 
                                           al5d_constants.POSITION_WRIST_ROTATION_MIN, 
                                           al5d_constants.POSITION_WRIST_ROTATION_MAX
        )
        # 5 gripper
        retval[5] = RobotHelper.map_ranges(self.gripper,
                                           al5d_constants.POSITION_GRIPPER_MIN,al5d_constants.POSITION_GRIPPER_MAX)
        return retval


    @staticmethod
    def from_normalized_vector(values):
        """Creates the rp from a normalized numpy vector"""
        rp = RobotPosition()
        # 0 height
        rp.height = RobotHelper.map_ranges(values[0], 0.0, 1.0, 
                                           al5d_constants.POSITION_HEIGHT_MIN, al5d_constants.POSITION_HEIGHT_MAX)
        # 1 distance
        rp.distance = RobotHelper.map_ranges(values[1], 0.0, 1.0,  
                                           al5d_constants.POSITION_DISTANCE_MIN, 
                                           al5d_constants.POSITION_DISTANCE_MAX)
        # 2 heading
        rp.heading = RobotHelper.map_ranges(values[2], 0.0, 1.0, 
                                           al5d_constants.POSITION_HEADING_MIN, 
                                           al5d_constants.POSITION_HEADING_MAX)
        # 3 wrist_angle
        rp.wrist_angle = RobotHelper.map_ranges(values[3], 0.0, 1.0, 
                                           al5d_constants.POSITION_WRIST_ANGLE_MIN, 
                                           al5d_constants.POSITION_WRIST_ANGLE_MAX)
        # 4 wrist_rotation
        rp.wrist_rotation = RobotHelper.map_ranges(values[4], 0.0, 1.0,  
                                           al5d_constants.POSITION_WRIST_ROTATION_MIN, 
                                           al5d_constants.POSITION_WRIST_ROTATION_MAX
        )
        # 5 gripper
        rp.gripper = RobotHelper.map_ranges(values[5], 0.0, 1.0, 
                                           al5d_constants.POSITION_GRIPPER_MIN,al5d_constants.POSITION_GRIPPER_MAX)
        return rp

    @staticmethod
    def from_vector(values):
        """Creates a RobotPosition from a numpy vector"""
        rp = RobotPosition()
        # 0 height
        rp.height = values[0]
        # 1 distance
        rp.distance = values[1]
        # 2 heading
        rp.heading = values[2]
        # 3 wrist_angle
        rp.wrist_angle = values[3]
        # 4 wrist_rotation
        rp.wrist_rotation = values[4]
        # 5 gripper
        rp.gripper = values[5]
        return rp

    def empirical_distance(self, other):
        """A weighted distance function between two robot positions"""
        w = np.ones([6]) / 6.0
        norm1 = np.array(self.to_normalized_vector())
        norm2 = np.array(other.to_normalized_vector())
        val = np.inner(w, np.abs(norm1 - norm2))
        return val



    def __str__(self):
        return f"Position: h={self.height:.2} dist={self.distance:.2} rot={self.heading:.2f} wa={self.wrist_angle:.2f} wrot={self.wrist_rotation:.2f} gripper={self.gripper:.2f}" 


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
    def ik_shoulder_elbow_wrist(target:RobotPosition):
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

    def move(self, target: RobotPosition):
        """Move to the specified target position: new version with one shot commands"""
        angle_z = 90 + target.heading
        angle_shoulder, angle_elbow, angle_wrist = self.ik_shoulder_elbow_wrist(target)
        angle_wrist_rotation = target.wrist_rotation
        # safety check here
        angles = np.zeros(5)
        angles[al5d_constants.SERVO_ELBOW] = angle_elbow
        angles[al5d_constants.SERVO_SHOULDER] = angle_shoulder
        angles[al5d_constants.SERVO_WRIST] = angle_wrist
        angles[al5d_constants.SERVO_WRIST_ROTATION] = angle_wrist_rotation
        angles[al5d_constants.SERVO_Z] = angle_z
        self.angle_controller.control_angles(angles, target.gripper)
        self.pos = target

