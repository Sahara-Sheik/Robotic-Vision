import numpy as np
from . import al5d_constants
from .al5d_helper import RobotHelper
from .al5d_pulse_controller import PulseController

class AngleController:
    """Implements a robot controller for the AL5D robot which performs control in the terms of angles (for the joints) and distance for the gripper.
    """

    def __init__(self, pulse_controller: PulseController):
        self.pulse_controller = pulse_controller
        self.positions = np.ones(self.pulse_controller.cnt_servos-1) * \
            RobotHelper.pulse_to_angle(
                self.pulse_controller.pulse_position_default)
        # FIXME: how do we set this?
        self.gripper_distance = 30

    def __str__(self):
        """Print the status of the robot"""
        return f"RobotAngleController positions = {self.positions} gripper={self.gripper_distance}"

    def as_dict(self):
        """Return the angles as a dictionary, for saving"""
        retval = {}
        for i, v in enumerate(self.positions):
            retval[i] = v
        return retval

    def control_servo_angle(self, servo, angle, speed=al5d_constants.CST_SPEED_DEFAULT):
        """Controls the servo through angle, by converting the angle to pulse. It sets the position assuming success. Works only for the 5 angle servos."""
        pulse, _ = RobotHelper.servo_angle_to_pulse(servo, angle)
        if servo < 0 or servo >= al5d_constants.SERVO_GRIP:
            raise Exception(f"Invalid servo for control_servo_angle {servo}")
        self.pulse_controller.control_servo_pulse(servo, pulse, speed)
        self.positions[servo] = angle

    def calculate_gripper(self, distance):
        """Calculates the pulse necessary to set the gripper to a certain 
        opening distance"""
        pulse = 1000 + 15 * (100 - distance)
        return pulse

    def control_gripper(self, distance, speed=al5d_constants.CST_SPEED_DEFAULT):
        """Sets the gripper to a certain opening distance [0..100]"""
        pulse = self.calculate_gripper(distance)
        self.pulse_controller.control_servo_pulse(al5d_constants.SERVO_GRIP, pulse, speed)
        self.gripper_distance = distance

    def control_angles(self, positions, gripper_distance):
        """Controls all the angles and the gripper in one shot"""
        target_pulses = np.zeros(self.pulse_controller.cnt_servos)
        for i in range(self.pulse_controller.cnt_servos - 1):
            target_pulses[i], _ = RobotHelper.servo_angle_to_pulse(i, positions[i])
        target_pulses[self.pulse_controller.cnt_servos-1] = self.calculate_gripper(gripper_distance)
        self.pulse_controller.control_pulses(target_pulses)
        self.positions = positions
        self.gripper_distance = gripper_distance
