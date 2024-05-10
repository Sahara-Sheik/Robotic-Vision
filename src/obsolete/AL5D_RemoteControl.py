from approxeng.input.selectbinder import ControllerResource
import time
import serial 

import cv2 as cv
import numpy as np
print(cv.__version__)

from AL5D_RobotController import RobotPosition, RobotPositionController
import logging
#logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)


# 2021-02: code for controlling the robot using the Xbox 360-type controller. 
# it integrates the video code

keys = {'axes': ['l', 'lt', 'lx', 'ly', 'r', 'rt', 'rx', 'ry'], 'buttons': ['circle', 'cross', 'dup', 'home', 'l2', 'r2', 'rs', 'select', 'square', 'start', 'triangle']}

class RobotRemoteControl:
    """A class describing the remote control of a robot using a 
    game controller"""

    def __init__(self):
        self.cap_1 = cv.VideoCapture(2)
        # self.cap.release()
        if not self.cap_1.isOpened():
            print("Cannot open camera")
            exit()
        self.rposc = RobotPositionController()
        self.shutdown_requested = False
        self.joystick = None
        self.target_position = None

    def update_from_joystick(self):
        """Updates the target_position field by reading the joystick buttons
        """
        presses = self.joystick.check_presses()
        if len(presses.buttons) > 0:
            print(presses.names)

        # lx is distance
        delta_distance = self.joystick['lx']
        self.target_position.position_distance += 0.1 * delta_distance
        # ly is height
        delta_height = self.joystick['ly']
        self.target_position.position_height += -0.1 * delta_height
        # rx is rotation 
        delta_rotation = self.joystick['rx']
        self.target_position.position_rotation += 3.0 * delta_rotation
        # ry is wrist angle
        delta_wrist_angle = self.joystick['ry']
        self.target_position.position_wrist_angle += -3.0 * delta_wrist_angle
        # panic button, the 
        if "select" in presses.names:
            print("Panic!")
            return RobotPosition()
        ## Gripper: buttons on the front
        if self.joystick["ls"] != None:
            value = min(5.0, self.joystick["ls"])
            print(value)
            if self.target_position.position_gripper < 100:
                self.target_position.position_gripper += 5 * value
        if self.joystick["rs"] != None:
            value = min(5.0, self.joystick["rs"])
            if self.target_position.position_gripper > 0:
                self.target_position.position_gripper -= 5 * value
        ## X and Y: gripper rotation
        if self.joystick["circle"] != None:
            value = min(5.0, self.joystick["circle"])
            if self.target_position.position_wrist_rotator < 90:
                self.target_position.position_wrist_rotator += 5 * value
        if self.joystick["triangle"] != None:
            value = min(5.0, self.joystick["triangle"])
            if self.target_position.position_wrist_rotator > -90:
                self.target_position.position_wrist_rotator -= 5 * value 
        ## Exit: back and start together
        if "home" in presses.names and "start" in presses.names:
            self.shutdown_requested = True
        return

    def shutdown_everything(self):
        print("Stopping robot in several seconds, please wait....")
        self.rposc.stop_robot()
        print("Stopping vision system")
        self.cap_1.release()
        cv.destroyAllWindows()    
        exit()

    def control_robot(self, callback = None):
        """Controls the robot using remote control"""
        try:
            with ControllerResource() as joystick:
                self.joystick = joystick
                print('Found a joystick and connected')
                print(self.joystick.controls)
                self.count = 0
                while self.joystick.connected:
                    self.count = self.count + 1
                    time.sleep(0.1)            
                    # Capture frame-by-frame
                    ret, self.camera_1 = self.cap_1.read()
                    # if frame is read correctly ret is True
                    if not ret:
                        logging.error("Can't receive frame (stream end?). Exiting ...")
                        break
                    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    # Display the resulting frame
                    cv.imshow('Camera 1', self.camera_1)
                    cv.waitKey(30) # wait for 30 milliseconds
                    self.target_position = self.rposc.get_position()
                    self.update_from_joystick()
                    if self.shutdown_requested:
                        break
                    self.rposc.move(self.target_position)
                    #
                    # We are calling the calledback
                    # 
                    if callback != None:
                        callback(self)            
        except IOError:
            # No joystick found, wait for a bit before trying again
            print('Unable to find any joysticks')
            #sleep(1.0)
        # FIXME: last callback to show that it is ok and maybe wrap up the commands
        self.shutdown_everything()

if __name__ == "__main__":
    rc = RobotRemoteControl()
    rc.control_robot()