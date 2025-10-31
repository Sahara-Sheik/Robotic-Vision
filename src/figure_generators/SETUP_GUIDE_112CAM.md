# 112 Camera Hemisphere Visualization Setup Guide

## 🎯 Complete Solution for Our 112 Camera System with our data

### Files Created for You:
1. **process_112_cameras.py** - Main processing script
3. **hemisphere_heatmap.py** - Core visualization library 
4. **real_data_visualizer.py** - Data loading utilities 

## 🚀 Quick Start Instructions

### Step 1: Update Your File Paths to where you have stored the mse results and camera postions
Open `process_112_cameras.py` and change these lines (staring line 362):

```python
# Change these to match your actual paths:
## Sahara's files"
BASE_FOLDER = "C:\...\112cam"
CAMERA_PLACEMENTS_FILE = "C:\\...\camera_placements.txt"
OUTPUT_DIR = "C:\....\\hemisphere_output"
```

### Step 2: Run the Script
```bash
cd C:\location of your files\
python process_112_cameras.py
```

## 📊 What We will Get

### 6 Hemisphere Visualizations
One for each component:
- **height** - How well each camera measures height
- **distance** - Distance measurement accuracy
- **heading** - Directional accuracy
- **wrist_angle** - Wrist angle precision
- **wrist_rotation** - Rotation measurement
- **gripper** - Gripper position accuracy

### 6 Hemisphere Visualizations for gradient
One for each component:
- **height** -
- **distance** 
- **heading** 
- **wrist_angle** 
- **wrist_rotation** 
- **gripper** 

### Output Files
```
hemisphere_output/
├── height_hemisphere_data.csv
├── height_hemisphere.png
├── distance_hemisphere_data.csv
├── distance_hemisphere.png
├── heading_hemisphere_data.csv
├── heading_hemisphere.png
├── wrist_angle_hemisphere_data.csv
├── wrist_angle_hemisphere.png
├── wrist_rotation_hemisphere_data.csv
├── wrist_rotation_hemisphere.png
├── gripper_hemisphere_data.csv
├── gripper_hemisphere.png
├── all_components_hemispheres.png  (6-panel combined view)  ### I NEED to fix this
└── component_summary.csv (statistical report)
```

## 🔍 Understanding Results

### Reading the Hemispheres
- **Red/Orange areas**: High accuracy (good camera positions)
- **Yellow areas**: Medium accuracy
- **Dark/Black areas**: Low accuracy (poor camera positions)
- **dots**: Individual camera positions

### Camera Layout
 112 cameras are arranged as:
- Ring at 0°: 12 cameras
- Ring at 10°: 12 cameras
- Ring at 20°: 12 cameras
- ... (continues every 10°)
- Ring at 80°: 12 cameras
- Top at 90°: 4 cameras

### How to run:
```python
# This reads our  112cam folder
python process_112_cameras.py
```

### Custom Analysis:
```python
from process_112_cameras import Camera112Processor

# Create processor
processor = Camera112Processor(
    base_folder="C:\\....\\112cam",
    camera_placements_file="C:\\...\camera_placements.txt"
)

# Process data
processor.parse_camera_placements()
processor.read_mse_values()

# Generate outputs
processor.generate_all_csvs("output")
processor.create_visualizations("output")
processor.generate_summary_report("output")
```

## 🛠️ Troubleshooting

### If MSE files are missing:
The script will use default values and warn us. folder structure needs to be as below or else we need to adjust the script:
```
112cam/
└── VisualProprioception_flow_00dev000/
    └── result/
        └── visual_proprioception/
            └── vp_comp_flow_all/
                └── msecomparison_values.csv
```

### If you get import errors:
Install required packages:
```bash
pip install numpy matplotlib scipy pandas
```

### If paths don't work:
Use raw strings in Python:
```python
BASE_FOLDER = r"C:\...\112cam"
```

## 📈 Interpreting Results

The summary report will show:
- **Best cameras** for each component
- **Mean accuracy** across all cameras
- **Standard deviation** showing consistency
- **Best/worst regions** on the hemisphere

We can use this to:
1. Identify optimal camera positions for specific tasks
2. Find blind spots in our coverage
3. Optimize camera placement for future setups

## ✅ Make sure we have this done before running

- [ ] All 112 camera folders exist (dev000 to dev111)
- [ ] Each has msecomparison_values.csv file
- [ ] camera_placements.txt has 112 camera positions
- [ ] Python packages installed (numpy, matplotlib, scipy, pandas)
- [ ] Paths updated in process_112_cameras.py




