from helper import ui_choose_task, ui_choose_demo, ui_edit_demo_metadata, analyze_demo
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def show(current, demo_dir, cameras, axs):
    """Show the current views for timestep. Also loads and returns the data"""
    print(f"Show the views of the current step {current}")
    for i, camera in enumerate(cameras):
        pic_file = Path(demo_dir, f"{current:05d}_{camera}.jpg")
        image = matplotlib.image.imread(pic_file)
        # this is not quite the stuff
        im = axs[i].imshow(image)
        im.set_data(image)
    plt.draw()

def main():
    print("======= Demonstration viewer and annotator ========")
    data_dir, task_dir = ui_choose_task()
    demo_dir = ui_choose_demo(task_dir)
    # Read the overall description
    file_overall = Path(demo_dir, "_demonstration.json")
    with open(file_overall, "r") as f:
        description = json.load(f)
    cameras, maxsteps = analyze_demo(demo_dir)

    fig, axs = plt.subplots(1, len(cameras), figsize=(6 * len(cameras),3))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    fig.show()

    current = 1
    while True:
        data_filename = Path(demo_dir, f"{current:05d}.json")
        with open(data_filename, "r") as f:
            data = json.load(f)
        print(f"Current {current} reward={data['reward']} annotation={data['annotation']}")
        show(current, demo_dir, cameras, axs)
        inp = input("<Enter>=next p: previous, r: reward, a: annotate [,]: trim x: exit >> ")
        if inp == "x":
            break
        if inp == "":
            current = min(maxsteps, current+1)
        if inp == "p":
            current = max(0, current-1)
        if inp == "r":
            rewardtext = input(f"change the current reward ({data['reward']}) to = ")
            reward = float(rewardtext)
            data["reward"] = reward
        if inp == "a":
            annotationtext = input("change the current annotation to:")
            data["annotation"] = annotationtext
        if inp == "[":
            description["trim-from"] = current
            print(f"Current demo trimmed from: {description['trim-from']}")
        if inp == "]":
            description["trim-to"] = current
            print(f"Current demo trimmed to: {description['trim-to']}")
        # this dumps to the last one, even if the current changed
        with open(data_filename, "w") as f:
            json.dump(data, f)
    ui_edit_demo_metadata(description)
    with open(file_overall, "w") as f:
        json.dump(description, f)

if __name__ == "__main__":
    main()