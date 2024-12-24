# Preprogrammed controller for the AL5D robot

from robot.al5d_position_controller import RobotPosition, PositionController

import time
# import serial 
from copy import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class ProgramController:
    """A programmed robot controller that works by reaching a set of waypoints with the robot. """

    def __init__(self, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        """Initialize the program controller, possibly connected to a real robot."""
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




    def set_waypoints(self, waypoints):
        """Sets the waypoints the robot needs to as list of positions"""
        self.waypoints = waypoints

    # FIXME: implement this

    def move_towards(current_height, target, max_velocity):
        if abs(target - current_height) <= max_velocity:
            # If the distance to the target is less than the max velocity, snap to target
            return target
        elif target > current_height:
            # Move up towards the target
            return current_height + max_velocity
        else:
            # Move down towards the target
            return current_height - max_velocity

    def move_to_waypoint(self):
        """FIXME: move towards the current waypoint with the specified speeds"""
        current = self.pos_target
        current_target = self.waypoints[0]
        # if we reached the one...
        if current.empirical_distance(waypoint[0]) > 0.1:
            return True, self.pos_target
        ## FIXME: do this for all
        heightdist = target.height - current.height
        target.height = current.height + (target.hei)

    # END FIXME implement this

    def control(self):
        """The main control loop"""
        self.exit_control = False
        while True:
            start_time = time.time() 
            key = self.camera_controller.update() 
            # if we are exiting, call the stopping of the robot, of the recording and the vision
            if self.exit_control:
                self.stop()
                break;
            
            # FIXME: the main step here has to be the setting of the next position. Basically we have to check whether we reached the next waypoint:
            reached, self.pos_target = self.move_to_waypoint()



            self.control_robot()
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 

    def control_robot(self):
        """Control the robot by sending a command to move towards the target
        FIXME: same as gamepad and keyboard
        """
        logger.info(f"Control robot: move to position {self.pos_target}")
        if self.robot_controller is not None:
            self.robot_controller.move(self.pos_target)
        logger.info("Control robot done.")            

    def stop(self):
        """Stops the controller and all the other subcomponents
        FIXME: same as gamepad, maybe it can be moved
        """
        self.robot_controller.stop_robot()
        if self.demonstration_recorder is not None:
            self.demonstration_recorder.stop()
        if self.camera_controller is not None:
            self.camera_controller.stop()

