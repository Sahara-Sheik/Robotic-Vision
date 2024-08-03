import sys
import cmd_demonstration_collector
import cmd_demonstration_viewer
import cmd_demonstration_replay
import cmd_learn_representation
import cmd_preprocess_representation
import cmd_learn_motion_controller
import cmd_run_robot

print("End-to-end trained robot control. v0.1.1 Aug 3, 2024, Lotzi Bölöni\n")

print("1. Collect demonstrations interactively")
print("2. View and annotate demonstrations")
print("3. Replay demonstrations on the robot")
print("4. Learn representations")
print("5. Preprocess representations")
print("6. Learn motion controller / robot policy")
print("7. Run the robot")

choice = int(input("Choose desired action: "))

if choice == 1:
    cmd_demonstration_collector.main()
    sys.exit(0)
if choice == 2:
    cmd_demonstration_viewer.main()
    sys.exit(0)
if choice == 3:
    cmd_demonstration_replay.main()
    sys.exit(0)
if choice == 4:
    cmd_learn_representation.main()
    sys.exit(0)
if choice == 5:
    cmd_preprocess_representation.main()
    sys.exit(0)
if choice == 6:
    cmd_learn_motion_controller.main()
    sys.exit(0)
if choice == 7:
    cmd_run_robot.main()
    sys.exit(0)

print(f"There is no such command {choice}!")