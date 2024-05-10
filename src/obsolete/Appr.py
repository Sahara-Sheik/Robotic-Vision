from approxeng.input.selectbinder import ControllerResource, ControllerNotFoundError
import time

try:
    with ControllerResource() as joystick:
        print('Found a joystick and connected')
        print(joystick.controls)
        while joystick.connected:
            presses = joystick.check_presses()
            if len(presses.buttons) > 0:
                print(presses.names)
except ControllerNotFoundError as e:
    print(e)
print("Bye")