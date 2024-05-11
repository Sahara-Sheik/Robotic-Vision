from . import al5d_constants
import numpy as np
import serial
import sys
import time
from .al5d_helper import RobotHelper
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PulseController:
    """A robot controller for the AL5D that operates at the level of the pulse. It interfaces directly to the hardware controller. Keeps the state of the robot in terms of the current pulse values of the servos. It is intended to be wrapped with more advanced controllers.

    https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/servo-erector-set-system/ses-electronics/ses-modules/ssc-32/ssc-32-manual/#qpwidth
    """

    def __init__(self, sp = None, device=None):
        self.cnt_servos = 6
        # the position taken by every motor when we are starting up
        # the robot
        self.pulse_position_default = 1500
        # the position taken by every motor when we are starting up
        # the robot
        # self.pulse_position_zero = 0
        # tracking the positions of the robot in terms of the pulse positions at the servos
        self.positions_pulse = np.ones(
            self.cnt_servos) * self.pulse_position_default
        if sp == None:
            if device == None:
                raise Exception("No device specified")                
            try:
                self.sp = serial.Serial(device, 9600)
            except serial.serialutil.SerialException as spex:
                print(spex)
                sys.exit(1)            
        else:
            self.sp = sp
        self.command_finished = True

    def as_dict(self):
        """Returns the pulse configuration as a dictionary, to be put into a saved json file"""
        retval = {}
        for i in range(self.cnt_servos):
            retval[i] = self.positions_pulse[i]
        return retval
            
    def start_robot(self, speed=100):
        """ Starts the robot and brings all the motors to the default position"""
        for servo in range(0, self.cnt_servos):
            self.control_servo_pulse(servo, self.pulse_position_default, speed)

    def stop_robot(self):
        """ Brings the robot relatively quickly to the default position. 
        Then, sets all motors to idle/unpowered (pulse = 0) """
        self.start_robot(1000)
        time.sleep(5)
        self.command_finished = False
        for servo in range(0, self.cnt_servos):
            command = f"#{servo} P{0}\r"
            self.sp.write(command.encode())

    def control_servo_pulse(self, servo, pulse, speed=al5d_constants.CST_SPEED_DEFAULT):
        """Sends a command to the particular servo to move to the particular
        pulse value, moving with the specified speed.
        """
        # TODO: some way to see if I passed a bad value - maybe a bad
        # value should not be passed???
        # TODO: option to wait here to execute???
        # FIXME: I removed this, because it does not play well with the constraints
        #pulse, _ = RobotControllerHelper.constrain(
        #    pulse, CST_PULSE_MIN, CST_PULSE_MAX)
        command = f"#{servo} P{pulse} S{speed}\r"
        logger.info(command)
        self.sp.write(command.encode())
        # FIXME: do we need this???
        self.wait_until_complete()
        self.positions_pulse[servo] = pulse

    def wait_until_complete(self):
        """Wait until all the commands are complete"""
        while True:
            command = "Q\r"
            self.sp.write(command.encode())
            readbytes = self.sp.read()
            # print(f"readbytes = {readbytes}")
            if readbytes == b'+':
                #print("command executed done")
                self.command_finished = True
                break
            self.command_finished = False

    def query_position(self, i):
        """Query the position of i"""
        command = f"QP {i}\r"
        self.sp.write(command.encode())
        time.sleep(0.1)
        readbyte = self.sp.read()
        val = 10 * ord(readbyte)
        return val


    def control_pulses(self, target_pulses, time=al5d_constants.TIME_DEFAULT):
        """Sends a command to set all the motors simultaneously"""
        command = ""
        for i in range(6):
            command += f"#{i} P {target_pulses[i]} "
        command += f" T{time}\r"
        logger.info(command)
        self.sp.write(command.encode())
        self.wait_until_complete()
        self.positions_pulse = target_pulses


    def control_servo_relative_pulse(self, servo, relative_pulse, speed=al5d_constants.CST_SPEED_DEFAULT):
        """Sends a command a particular servo to change its position with 
        the specified pulse value (positive or negative), and a specific speed
        """
        # TODO: some way to check whether a bad value was passed, probably error should be raised.
        # TODO: option to wait here to execute
        pulse = self.positions_pulse[servo] + relative_pulse
        pulse, constrained = RobotHelper.constrain(
            pulse, al5d_constants.CST_PULSE_MIN, al5d_constants.CST_PULSE_MAX)
        if constrained:
            logger.warning("out of range at control_servo_relative_pulse")
        self.control_servo_pulse(servo, pulse, speed)

    def __str__(self):
        """Print the status of the robot"""
        return f"AL5D_PulseController positions = {self.positions_pulse}"
