import pathlib
import logging
import platform

# Setting up the paths for the robotic imitation learning projects

project = "2021-10-03-RobotImitationLearning"

name = platform.node()
if name == "tredy2":
    root_dir = pathlib.Path("/home/lboloni/Documents/Hackingwork")
elif name == "kutya":
    root_dir = pathlib.Path("/Users/lboloni/Documents/Work")
root_temp_dir = pathlib.Path(root_dir, "__Temporary")
project_dir = pathlib.Path(root_dir, project)
if not project_dir.exists():
    raise Exception(f"Project directory {project_dir} does not exist.")
# data directory
data_dir = pathlib.Path(project_dir, "data")
unsupervised_dir = pathlib.Path(data_dir, "images-32-task-3001-10090")
demonstration_dir = pathlib.Path(data_dir, "demonstration-32-task-3001-10090")

# create the temporary directory, for saving a cached dataset, snapshots etc
temp_dir = pathlib.Path(root_temp_dir, project)
temp_dir.mkdir(exist_ok = True, parents = True)
# visual model path
visual_module_model_path = pathlib.Path(temp_dir, "cvae")

# mpl model path
controller_model_path = pathlib.Path(temp_dir, "mlp")

# demonstration path - normally this should not be in temporary, but for the time being, ok
demonstrations_path = pathlib.Path(temp_dir, "demonstrations")

# Rouhollah's full collected data
project_rouhi = "2021-05-01-Rouhollah-Code-and-Data"
project_rouhi_dir = pathlib.Path(root_dir, project_rouhi)
demonstrations_rouhi_dir = pathlib.Path(project_rouhi_dir, "trajectories", "al5d-32")


# set the logging to print everything
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)