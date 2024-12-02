import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


from helper import ui_choose_task
from gamepad.gamepad_controller import GamepadController
from keyboard.keyboard_controller import KeyboardController
from robot.al5d_position_controller import PositionController
from camera.camera_controller import CameraController


from sensorprocessing import sp_conv_vae
from behavior_cloning.bc_LSTM import LSTMXYPredictor, LSTMResidualController


from settings import Config

def main():
    print("======= Run the robot with the behavior from settings =========")

    print("... Initializing the robot")

    # the robot position controller
    robot_controller = PositionController(Config()["robot"]["usb_port"]) 

    print("... Initializing the camera")

    img_size = Config()["robot"]["saved_image_size"]
    # (256, 256) # was (128, 128)
    camera_controller = CameraController(devices = Config()["robot"]["active_camera_list"], img_size = img_size)
    camera_controller.visualize = True
    lead_camera = next(iter(camera_controller.capture_devs))
    print(f"Lead camera: {lead_camera}")

    print("... Initialize and load the image processing")
    sp = sp_conv_vae.ConvVaeSensorProcessing()

    print("...Initialize and load the controller (behavior cloning)")

    # Original
    latent_size = Config()["robot"]["latent_encoding_size"]  
    hidden_size = 32  # degrees of freedom in the robot
    output_size = 6  # degrees of freedom in the robot
    num_layers = 2

    # Instantiate model, loss function, and optimizer
    model = LSTMXYPredictor(latent_size=latent_size, hidden_size=hidden_size, output_size = output_size, num_layers=num_layers)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    filename_lstm = Config()["controller"]["lstm_model_file"]
    model.load_state_dict(torch.load(filename_lstm))
    print("...BC controller initialized")


    while True:
        print("capture the image")
        camera_controller.update()
        image = camera_controller.images[lead_camera]
        print("process the image")
        z = sp.process(image)
        print(z)
        print("generate the action")

        input = torch.from_numpy(z[i])
        input = input.unsqueeze(0)
        input = input.unsqueeze(0)
        print(input)
        a_pred = model.forward_keep_state(input)
        print(f"a_pred: {a_pred}")
        print("verify if the action is acceptable")
        response = input(f"The action is {a_pred}. Is it acceptible? (y/n)")
        if response == "y":
            print("Enact response")
            # FIXME: probably the action must be passsed on a different format
            robot_controller.move(a_pred)
            print("FIXME: not implemented yet")
        else:
            print("Action not allowed, terminating")
            break
    print("Shutting down robot")
    robot_controller.stop_robot()
    print("End of 'cmd_run_robot'")

if __name__ == "__main__":
    main()