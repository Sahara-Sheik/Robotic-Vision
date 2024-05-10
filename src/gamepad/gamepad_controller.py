from robot.al5d_position_controller import RobotPosition, PositionController
from approxeng.input.selectbinder import ControllerResource, ControllerNotFoundError
import time
# import serial 
from copy import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class GamepadController:
    """
    A controller to control an AL5D robot with an X360 type controller (tested with the Voyee). This controller is based on the approxeng.input library and it is only working in Linux.
    """
    def __init__(self, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        """Initialize the XBox controller, possibly connected to a real robot."""
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
        # the velocities corresponding to maximum push
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
        try:
            with ControllerResource() as joystick:
                print('Found a joystick and connected')
                print(joystick.controls)
                self.exit_control = False
                while True:
                    if not joystick.connected:
                        break
                    start_time = time.time() 
                    self.poll_controller(joystick) 
                    # if we are exiting, call the stopping of the robot, of the recording and the vision
                    if self.exit_control:
                        self.stop()
                        break;
                    self.control_robot() 
                    self.update()
                    end_time = time.time() 
                    execution_time = end_time - start_time 
                    self.last_interval = execution_time
                    time_to_sleep = max(0.0, self.controller_interval - execution_time) 
                    time.sleep(time_to_sleep) 
        except ControllerNotFoundError as e:
            print(e)
            print("Bye")
        finally:
            print("Done")
            self.robot_controller.stop_robot()


    def stop(self):
        """Stops the controller and all the other subcomponents"""
        self.robot_controller.stop_robot()
        if self.demonstration_recorder is not None:
            self.demonstration_recorder.stop()
        if self.camera_controller is not None:
            self.camera_controller.stop()

    def update(self):
        """Updates the state of the various components"""
        logger.info(f"Update started")
        if self.camera_controller is not None:
            self.camera_controller.update()
        if self.demonstration_recorder is not None:
            self.demonstration_recorder.save()
        logger.info(f"Update done")
        

    def poll_controller(self, joystick):
        """Polls what is going on with the controller and updates the target position of the robot accordingly. """
        # this checks the new presses, but maybe we want some of the buttons
        # to keep doing if they are kept down
        presses = joystick.check_presses()
        if len(presses.buttons) > 0:
            print(presses.names)
            # print(joystick.lx)
        # calculate the changes with specific velocities
        # distance: left-right on the left joystick
        delta_distance = joystick.lx * self.v_distance * self.last_interval
        # height: up-down on on the left joystick
        delta_height = joystick.ly * self.v_height * self.last_interval
        # rotation: left-right on the right joystick
        delta_heading = joystick.rx * self.v_heading * self.last_interval
        # wrist angle: up-down on the right joystick
        delta_wrist_angle = joystick.ry * self.v_wrist_angle * self.last_interval
        # wrist rotation: the two triggers
        delta_wrist_rotation = (joystick.lt - joystick.rt) 
        # gripper open-close: left 
        delta_gripper = 0
        # the triggers immediately open and close the gripper
        # the pad opens/closes it gradually
        if "l1" in presses.names:
            delta_gripper = 100
        if "r1" in presses.names:
            delta_gripper = -100
        if joystick["dleft"] is not None: # if held, returns the seconds
            delta_gripper += self.v_gripper * self.last_interval
            logging.warning(f"{joystick.dleft}")
        if joystick["dright"] is not None: 
            delta_gripper -= self.v_gripper * self.last_interval


        # square aka x - exit control
        if "square" in presses.names:
            self.exit_control = True
            return
        # home  
        if "home" in presses.names:
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

    def control_robot(self):
        """Control the robot by sending a command to move towards the target"""
        logger.info(f"Control robot: move to position {self.pos_target}")
        if self.robot_controller is not None:
            self.robot_controller.move(self.pos_target)
        logger.info("Control robot done.")

    def __str__(self):
        return f"XBoxController:\n\t{self.pos_target}"