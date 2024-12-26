from helper import ui_choose_task, ui_choose_demo, analyze_demo
from pathlib import Path
import json
import time
from robot.al5d_position_controller import PositionController, RobotPosition    
from settings import Config

def replay_full_speed(robot_controller, demo_dir, maxsteps, delay = 0.1):
    """Runs"""
    # the robot position controller
    for i in range(1, maxsteps):
        file_name = Path(demo_dir, f"{i:05d}.json")
        with open(file_name, "r") as f:
            data = json.load(f)
        position = RobotPosition(data["rc-position-target"])
        robot_controller.move(position)
        time.sleep(delay)

def replay_step_by_step(robot_controller, demo_dir, maxsteps, delay = 0.1):
    current = 1
    while True:
        file_name = Path(demo_dir, f"{current:05d}.json")
        with open(file_name, "r") as f:
            data = json.load(f)
        robot_controller.move(data["rc-position-target"])        
        inp = "Run next (<Enter>) Run next 10 (n) Go back (b) Beginning (1), Go to end (G), Exit (x)"
        if inp == "x":
            break
        if inp == "":
            current = min(current+1, maxsteps)
        if inp == "b":
            current = max(0, current - 1)
        if inp == "0":
            current = 1
        if inp == "G":
            current = maxsteps
        if inp == "n":
            current  = min(current + 10, maxsteps)



def main():
    print("======== Demonstration replay =========")

    print("Connecting to the robot (make sure it is on, and connected)")
    robot_controller = PositionController(Config().values["robot"]["usb_port"]) 
    print("Connection to robot successful")

    data_dir, task_dir = ui_choose_task()
    demo_dir = ui_choose_demo(task_dir)
    cameras, maxsteps = analyze_demo(demo_dir)
    while True:
        inp = input("Replay (r), Replay-step-by-step (s) Exit (x): ")
        if inp == "r":
            replay_full_speed(robot_controller, demo_dir, maxsteps, delay = 0.1)
        if inp == "s":
            replay_step_by_step()
        if inp == "x":
            print("Shutting down the robot")
            robot_controller.stop_robot()
            print("Robot shot down.")
            break

if __name__ == "__main__":
    main()