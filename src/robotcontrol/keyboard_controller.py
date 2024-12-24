# Keyboard based controller for the AL5D robot

from robot.al5d_position_controller import RobotPosition, PositionController

import time
# import serial 
from copy import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class KeyboardController:
    """
    A controller to control an AL5D robot with the keyboard. The assumption is that the keys are read out by the OpenCV camera controller.

    These are the keys for typical FPS games
    
    WSAD for move.
    Ctrl for crouch or sneak.
    Left click for primary attack, Right click for secondary attack.
    Shift for run.
    E or F for "activate"
    "R" for reload.

    The keys to be used here:
    W 
    S
    A 
    D 
    up 82
    down 84
    left  81
    right 83
    Q
    pgup 85
    pgdown 86
    left-shift  225
    right-shift  226
    left-alt 233
    right-alt 234
    """

    def __init__(self, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        """Initialize the keyboard controller, possibly connected to a real robot."""
        self.robot_controller = robot_controller
        if robot_controller is not None:
            self.pos_current = self.robot_controller.get_position()
        else:
            self.pos_current = RobotPosition()
        self.pos_target = copy(self.pos_current)
        # the home position
        self.pos_home = copy(self.pos_current)
        # the interval at which we are polling the controller
        self.controller_interval = 0.1
        # the actual interval
        self.last_interval = self.controller_interval
        # the interval at which we are updating the robot
        self.robot_interval = 0.1
        # the velocities of movements
        self.v_distance = 1.0 # inch / second
        self.v_height = 1.0 # inch / second
        self.v_heading = 15.0 # angle degree / second
        self.v_gripper = 50.0 # percentage / second
        self.v_wrist_angle = 15.0 # angle degree / second
        self.v_wrist_rotator = 0.1 # angle degree / second
        self.camera_controller = camera_controller
        self.demonstration_recorder = demonstration_recorder

    def control(self):
        """The main control loop"""
        self.exit_control = False
        while True:
            start_time = time.time() 
            key = self.camera_controller.update() 
            self.process_key(key)
            # if we are exiting, call the stopping of the robot, of the recording and the vision
            if self.exit_control:
                self.stop()
                break;
            print(key)
            self.control_robot()
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 

    def process_key(self, key):
        """Sets the target location based on the key pressed"""
        keycode = key & 0xFF
        # distance: s and a 
        delta_distance = 0
        if keycode == ord('s'): # forward
            delta_distance = self.v_distance * self.last_interval
        if keycode == ord('a'): # backward
            delta_distance = - self.v_distance * self.last_interval
        # height: w and z
        delta_height = 0
        if keycode == ord("w"): # up
            delta_height = self.v_height * self.last_interval
        if keycode == ord("z"): # down
            delta_height = - self.v_height * self.last_interval
        # rotation: left-right FIXME: maybe this should go on pgup 85/86
        delta_heading = 0
        if keycode == 81: # left -> rotate-left
            delta_heading = self.v_heading * self.last_interval
        if keycode == 83: # right -> rotate-right
            delta_heading = - self.v_heading * self.last_interval

        # wrist angle: pg-up pg-down FIXME: maybe this should go on up down
        delta_wrist_angle = 0
        if keycode == 85: # pgup - wrist angle up
            delta_wrist_angle = self.v_wrist_angle * self.last_interval 
        if keycode == 86: # pgdn - wrist angle down
            delta_wrist_angle = - self.v_wrist_angle * self.last_interval 

        # wrist rotation: left-right
        delta_wrist_rotation = 0
        if keycode == 81: # left --> wrist-rotate-left
            delta_wrist_rotation = self.v_wrist_rotator * self.last_interval 
        if keycode == 83: # right --> write-rotate-right
            delta_wrist_rotation = - self.v_wrist_rotator * self.last_interval 

        # gripper open-close: right alt / shift 226 / 234
        delta_gripper = 0
        # the right alt/shift immediately closes and opens the gripper
        if keycode == 226:
            delta_gripper = 100
        if keycode == 234:
            delta_gripper = -100
        # the left alt/shift opens/closes it gradually 225/233
        if keycode == 225:
            delta_gripper += self.v_gripper * self.last_interval
        if keycode == 233:
            delta_gripper += - self.v_gripper * self.last_interval

        # square aka x - exit control
        if keycode == ord("x"):
            self.exit_control = True
            return
        # home h  
        if keycode == ord("h"):
            self.pos_target = copy(self.pos_home)
            return
        # applying the changes 
        self.pos_target.distance += delta_distance
        self.pos_target.height += delta_height
        self.pos_target.heading += delta_heading
        self.pos_target.wrist_angle += delta_wrist_angle
        self.pos_target.wrist_rotation += delta_wrist_rotation
        self.pos_target.gripper += delta_gripper
        # FIXME: applying a safety reset which prevents us going out of range
        ok, self.pos_target = RobotPosition.limit(self.pos_target)
        if not ok:
            logger.warning(f"DANGER! exceeded range! {self.pos_target}")
        logger.warning(f"Target: {self.pos_target}")

    def stop(self):
        """Stops the controller and all the other subcomponents
        FIXME: same as gamepad, maybe it can be moved
        """
        self.robot_controller.stop_robot()
        if self.demonstration_recorder is not None:
            self.demonstration_recorder.stop()
        if self.camera_controller is not None:
            self.camera_controller.stop()

    def update(self):
        """Updates the state of the various components
        FIXME: same as gamepad, maybe it can be moved
        """
        logger.info(f"Update started")
        if self.camera_controller is not None:
            self.camera_controller.update()
        if self.demonstration_recorder is not None:
            self.demonstration_recorder.save()
        logger.info(f"Update done")        

    def control_robot(self):
        """Control the robot by sending a command to move towards the target
        FIXME: same as gamepad, maybe it can be moved
        """
        logger.info(f"Control robot: move to position {self.pos_target}")
        if self.robot_controller is not None:
            self.robot_controller.move(self.pos_target)
        logger.info("Control robot done.")