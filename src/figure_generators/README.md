# Hemisphere Camera Accuracy Heatmap Visualization

This program creates comprehensive heatmap visualizations showing camera position accuracy on a hemisphere, with gradient field analysis.

## Files Generated

### Core Python Scripts
- **hemisphere_heatmap.py**: Main visualization class with all core functionality
- **real_data_visualizer.py**: Extended version that can load real data from CSV files

### Output Files
- **hemisphere_analysis.png**: Static 3-view visualization showing:
  - 3D hemisphere with accuracy heatmap and gradient arrows
  - Top-down circular heatmap view
  - Gradient magnitude visualization

- **interactive_hemisphere.html**: Interactive 3D visualization (open in browser)
- **sample_camera_data.csv**: Example input data format
- **camera_analysis_results.csv**: Detailed analysis results sorted by accuracy

## Features

### 1. Camera Position Generation
- **Uniform distribution**: Using Fibonacci spiral for even coverage
- **Random distribution**: Randomly placed cameras
- **Clustered distribution**: Cameras grouped in clusters

### 2. Accuracy Pattern Types
- **Gradient**: Linear gradient across hemisphere
- **Hotspots**: Multiple high-accuracy zones
- **Radial**: Accuracy decreases from center
- **Sinusoidal**: Wave-like patterns

### 3. Visualizations
- **3D Hemisphere**: Shows accuracy as color with optional gradient arrows
- **Top-Down View**: Circular heatmap with contour lines
- **Gradient Magnitude**: Shows rate of accuracy change
- **Interactive 3D**: Rotatable, zoomable HTML visualization

## Usage

### Basic Usage (Synthetic Data)
```python
from hemisphere_heatmap import HemisphereHeatmap

# Create visualizer for our case for now 112 but we can change this
vis = HemisphereHeatmap(n_cameras=100)

# Generate camera positions
vis.generate_camera_positions(distribution='uniform')

# Generate accuracy values
vis.generate_accuracy_values(pattern='hotspots')

# Create visualization
vis.plot_hemisphere_heatmap(
    show_cameras=True,
    show_gradient=True,
    colormap='RdYlBu_r'
)
```

### Using Real Data
```python
from real_data_visualizer import RealDataHemisphereVisualizer

# Create visualizer
vis = RealDataHemisphereVisualizer()

# Load data from CSV
vis.load_data_from_csv('your_camera_data.csv')

# Generate visualizations
vis.plot_hemisphere_heatmap()
vis.create_interactive_plot()
```

### CSV Data Format

#### Cartesian Format (x, y, z, accuracy)
```csv
x,y,z,accuracy
0.5,0.3,0.8,0.95
-0.2,0.7,0.6,0.82
...
```

#### Spherical Format (theta, phi, radius, accuracy)
```csv
theta,phi,radius,accuracy
0.5,1.2,1.0,0.89
0.8,2.1,1.0,0.75
...
```

## Key Algorithms

### 1. Accuracy Interpolation
- Uses Radial Basis Function (RBF) interpolation
- Gaussian kernel with smoothing for continuous surface
- Creates smooth transitions between sparse camera points

### 2. Gradient Computation
- Numerical gradients computed on interpolated mesh
- Arrows show direction of maximum accuracy increase
- Magnitude indicates rate of change

### 3. Hemisphere Mesh Generation
- Parametric surface using spherical coordinates
- Adjustable resolution for quality vs. performance
- Proper aspect ratio for realistic hemisphere shape

## Customization Options

### Color Maps
- 'RdYlBu_r': Red-Yellow-Blue reversed (default)
- 'viridis': Purple to yellow
- 'plasma': Purple to pink to yellow
- 'coolwarm': Blue to red
- 'seismic': Blue to white to red

### Visualization Parameters
- `show_cameras`: Display actual camera positions
- `show_gradient`: Display gradient vector field
- `resolution`: Mesh resolution (default 50)
- `save_path`: Output file location

## Analysis Report
The program automatically generates statistics including:
- Mean, min, max accuracy values
- Standard deviation
- Best/worst camera positions
- Position coordinates in both Cartesian and spherical

## Requirements
- Python 3.7+
- numpy
- matplotlib
- scipy
- pandas
- plotly (optional, for interactive plots)

## Installation
```bash
pip install numpy matplotlib scipy pandas plotly
```

## Example Workflow

1. **Generate sample data**:
   ```bash
   python real_data_visualizer.py
   ```

2. **Use your own data**:
   ```bash
   python real_data_visualizer.py your_data.csv
   ```

3. **View outputs**:
   - Open `hemisphere_analysis.png` for static visualization
   - Open `interactive_hemisphere.html` in browser for interactive view
   - Check `camera_analysis_results.csv` for detailed metrics

## Understanding the Visualizations

### Heatmap Colors
- **Red/Yellow**: High accuracy regions
- **Blue**: Low accuracy regions
- **Contour lines**: Lines of equal accuracy

### Gradient Arrows
- **Direction**: Points toward increasing accuracy
- **Length**: Proportional to rate of change
- **Density**: Sampled grid points for clarity

### Camera Markers
- **Color**: Matches accuracy value
- **Size**: Uniform or can be scaled by importance
- **Border**: Black edge for visibility

## Tips for Best Results

1. **Data Quality**: Ensure camera positions are on hemisphere surface
2. **Coverage**: Aim for good spatial distribution of cameras
3. **Normalization**: Keep accuracy values in [0, 1] range
4. **Resolution**: Higher mesh resolution = smoother visualization but slower
5. **Interpolation**: Adjust smoothing parameter for different effects

## Troubleshooting

- **Memory issues**: Reduce mesh resolution
- **Slow rendering**: Decrease number of gradient arrows
- **Poor interpolation**: Adjust RBF epsilon and smoothing parameters
- **Missing cameras**: Check CSV format and coordinate system



