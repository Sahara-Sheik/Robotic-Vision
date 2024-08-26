import sys

print("Vision-based robot manipulator.\n v0.1.2 Aug 26, 2024, Lotzi Bölöni\n")

print("1. Collect demonstrations interactively")
print("2. View and annotate demonstrations")
print("3. Replay demonstrations on the robot")
print("4. Learn representations")
print("5. Preprocess representations")
print("6. Learn motion controller / robot policy")
print("7. Run the robot")

choice = int(input("Choose desired action: "))

if choice == 1:
    import cmd_demonstration_collector
    cmd_demonstration_collector.main()
    sys.exit(0)
if choice == 2:
    import cmd_demonstration_viewer
    cmd_demonstration_viewer.main()
    sys.exit(0)
if choice == 3:
    import cmd_demonstration_replay
    cmd_demonstration_replay.main()
    sys.exit(0)
if choice == 4:
    import cmd_learn_representation
    cmd_learn_representation.main()
    sys.exit(0)
if choice == 5:
    import cmd_preprocess_representation
    cmd_preprocess_representation.main()
    sys.exit(0)
if choice == 6:
    import cmd_learn_motion_controller
    cmd_learn_motion_controller.main()
    sys.exit(0)
if choice == 7:
    import cmd_run_robot
    cmd_run_robot.main()
    sys.exit(0)

print(f"There is no such command {choice}!")