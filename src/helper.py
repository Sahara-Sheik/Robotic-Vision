from pathlib import Path
import sys
from settings import Config

# shared code for the various commands

def get_demodir():
    """Returns the data directory associated with this project"""
    demodir = Path(Config().values["demos"]["directory"])
    demodir.mkdir(parents=True, exist_ok=True)
    return demodir

def ui_choose_task(offer_task_creation = False):
    data_dir = get_demodir()
    demos_dir = Path(data_dir, "demos")
    if demos_dir.exists():
        tasks = [item for item in demos_dir.iterdir() if item.is_dir()]
        tasks = sorted(tasks, key=lambda x: x.name)
        print(f"Demo directory {demos_dir} found with tasks {tasks}")
    else:
        print(f"Demo directory {demos_dir} was not found.")
        proceed = input("Create? (y/N)")
        if proceed == "y":
            demos_dir.mkdir(parents=True, exist_ok = True)
        else:
            sys.exit(0)
    # Choose the task
    tasks_dict = {}
    for i, t in enumerate(tasks):
        tasks_dict[i] = t
    for key in tasks_dict:
        print(f"\t{key}: {tasks_dict[key].name}")
    task_id = int(input("Choose the task: "))
    task_dir = tasks_dict[task_id]
    print(f"You chose task: {task_dir.name}")
    return data_dir, task_dir

def ui_choose_demo(task_dir):
    """Chooses a demonstration directory from a task"""
    tasks = [item for item in task_dir.iterdir() if item.is_dir()]
    print(f"Demo directory {task_dir} found with demonstrations {tasks}")
    demos_dict = {}
    for i, t in enumerate(tasks):
        demos_dict[i] = t
    for key in demos_dict:
        print(f"\t{key}: {demos_dict[key].name}")
    demo_id = int(input("Choose the demonstration: "))
    demo_dir = demos_dict[demo_id]
    print(f"You chose demonstration: {demo_dir.name}")
    return demo_dir

def print_demo_description(description):
    print(f"Current values for demonstration {description['name']} and {description['task']}")
    for key in sorted(description.keys()):
        print(f"{key} = {description[key]}")


def ui_edit_demo_metadata(description):
    """Interactively collects generic labels for the demonstrations"""
    while True:
        print_demo_description(description)
        val = input("s+/s-: success, q: quality, t: text-annotation x: exit")
        if val == "s+":
            description["success"] = True
        if val == "s-":
            description["success"] = False
        if val == "q":
            qstring = input("Quality value (between 0.0 and 1.0):")
            qval = float(qstring)
            if (qval >= 0.0) and (qval <= 1.0):
                description["quality"] = qval
            else:
                print(f"incorrect quality value {qval}")
        if val == "t": 
            annotation = input("text annotation of the over demo")
            description["text-annotation"] = annotation
        if val == "x":
            break

def analyze_demo(demo_dir):
    """
    Analize a demo directory, find the cameres etc. 
    FIXME: maybe easier to just write this in the _demonstration.json.
    """
    maxsteps = -1
    cameraset = {}
    for a in demo_dir.iterdir():
        if a.name.endswith(".json") and a.name.startswith("0"):
            count = int(a.name.split(".")[0])
            maxsteps = max(maxsteps, count)
        if a.name.endswith(".jpg"):
            cameraname = a.name[6:-4]
            cameraset[cameraname] = cameraname
    cameras = sorted(cameraset.keys())
    print(f"Cameras found: {cameras}")
    print(f"There are {maxsteps} steps in this demonstration")
    print(f"This demonstration was recorded by the following cameras: {cameras}")
    return cameras, maxsteps
