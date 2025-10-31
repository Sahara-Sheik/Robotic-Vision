import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hemisphere_heatmap import HemisphereHeatmap

class RealDataHemisphereVisualizer(HemisphereHeatmap):
    """Extended visualizer"""

    def load_data_from_csv(self, filepath):
        """
        Load camera positions and accuracy from CSV file.

        Expected CSV format:
        x, y, z, accuracy
        0.5, 0.3, 0.8, 0.95
        ...
        """
        df = pd.read_csv(filepath)

        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            self.camera_positions = df[['x', 'y', 'z']].values
            self.n_cameras = len(self.camera_positions)
        else:
            raise ValueError("CSV must contain 'x', 'y', 'z' columns")

        if 'accuracy' in df.columns:
            self.accuracy_values = df['accuracy'].values
        else:
            print("No 'accuracy' column found. Generating synthetic values.")
            self.generate_accuracy_values()

        return self.camera_positions, self.accuracy_values

    def load_data_from_spherical(self, filepath):
        """
        Load camera positions from spherical coordinates.

        Expected CSV format:
        theta, phi, radius, accuracy
        0.5, 1.2, 1.0, 0.89
        ...
        """
        df = pd.read_csv(filepath)

        if 'theta' in df.columns and 'phi' in df.columns:
            theta = df['theta'].values
            phi = df['phi'].values
            radius = df['radius'].values if 'radius' in df.columns else np.ones(len(theta))

            # Convert to Cartesian
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            self.camera_positions = np.column_stack([x, y, z])
            self.n_cameras = len(self.camera_positions)

            if 'accuracy' in df.columns:
                self.accuracy_values = df['accuracy'].values
            else:
                print("No 'accuracy' column found. Generating synthetic values.")
                self.generate_accuracy_values()
        else:
            raise ValueError("CSV must contain 'theta' and 'phi' columns")

        return self.camera_positions, self.accuracy_values

    def export_results(self, output_path='camera_analysis_results.csv'):
        """
        Export the camera positions, accuracy values, and analysis results.
        """
        if self.camera_positions is None or self.accuracy_values is None:
            print("No data to export!")
            return

        # Create dataframe
        df = pd.DataFrame({
            'camera_id': range(self.n_cameras),
            'x': self.camera_positions[:, 0],
            'y': self.camera_positions[:, 1],
            'z': self.camera_positions[:, 2],
            'accuracy': self.accuracy_values
        })

        # Add spherical coordinates
        r = np.sqrt(np.sum(self.camera_positions**2, axis=1))
        theta = np.arccos(self.camera_positions[:, 2] / r)
        phi = np.arctan2(self.camera_positions[:, 1], self.camera_positions[:, 0])

        df['radius'] = r
        df['theta'] = theta
        df['phi'] = phi

        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

        return df

    def create_interactive_plot(self):
        """
        Create an interactive version using plotly for better exploration.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install it with: pip install plotly")
            return

        # Create hemisphere mesh
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)

        # Interpolate accuracy
        accuracy_mesh = self.interpolate_accuracy(X, Y, Z)

        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter3d'}]],
            subplot_titles=['Accuracy Heatmap', 'Camera Positions']
        )

        # Add surface plot
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=accuracy_mesh,
                colorscale='RdYlBu_r',
                showscale=True,
                name='Accuracy',
                colorbar=dict(x=0.45, len=0.5)
            ),
            row=1, col=1
        )

        # Add camera positions
        fig.add_trace(
            go.Scatter3d(
                x=self.camera_positions[:, 0],
                y=self.camera_positions[:, 1],
                z=self.camera_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.accuracy_values,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(x=1.02, len=0.5)
                ),
                text=[f'Camera {i}<br>Accuracy: {acc:.3f}'
                      for i, acc in enumerate(self.accuracy_values)],
                hovertemplate='%{text}<extra></extra>',
                name='Cameras'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title='Interactive Hemisphere Accuracy Visualization',
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene2=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=False,
            width=1200,
            height=600
        )
        #####################################################################
        ########          Sahara's path change this             #############
        #####################################################################
        # Save and show
        fig.write_html('/Users/rkhan/Downloads/interactive_hemisphere.html')
        print("Interactive plot saved to interactive_hemisphere.html")

        return fig


def generate_sample_data(n_cameras=100, output_file='sample_camera_data.csv'):
    """
    Generate sample camera data file for testing.
    """
    # Generate positions on hemisphere
    indices = np.arange(0, n_cameras, dtype=float) + 0.5
    theta = np.arccos(1 - indices/n_cameras)
    phi = np.pi * (1 + 5**0.5) * indices

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Generate accuracy with some pattern
    # Higher accuracy near the top and in certain regions
    accuracy = 0.5 + 0.3 * z  # Base accuracy increases with height

    # Add hotspots
    hotspot1 = np.array([0.5, 0.5, 0.7])
    hotspot2 = np.array([-0.5, 0.5, 0.7])

    for i in range(n_cameras):
        pos = np.array([x[i], y[i], z[i]])
        dist1 = np.linalg.norm(pos - hotspot1)
        dist2 = np.linalg.norm(pos - hotspot2)
        accuracy[i] += 0.3 * np.exp(-3 * dist1**2)
        accuracy[i] += 0.3 * np.exp(-3 * dist2**2)

    # Add noise
    accuracy += np.random.normal(0, 0.05, n_cameras)
    accuracy = np.clip(accuracy, 0, 1)

    # Create dataframe
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'accuracy': accuracy
    })

    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")

    return df


#  usage script
if __name__ == "__main__":
    import sys

    print("Hemisphere Camera Accuracy Visualization")
    print("="*50)

    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Loading data from: {data_file}")

        visualizer = RealDataHemisphereVisualizer()

        #load the data
        try:
            if data_file.endswith('.csv'):
                # Check if it's spherical or Cartesian
                df_check = pd.read_csv(data_file)
                if 'theta' in df_check.columns and 'phi' in df_check.columns:
                    visualizer.load_data_from_spherical(data_file)
                else:
                    visualizer.load_data_from_csv(data_file)
            else:
                print("Unsupported file format. Please use CSV.")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    else:
        print("No data file provided. Generating sample data...")

        # Generate sample data
        generate_sample_data(n_cameras=100, output_file="C:\\Users\\rkhan\\Downloads\\file\\sample_camera_data.csv")
        # Create visualizer and load the sample data
        visualizer = RealDataHemisphereVisualizer()
        visualizer.load_data_from_csv("C:\\Users\\rkhan\\Downloads\\file\\sample_camera_data.csv")

    # Generate report
    visualizer.generate_report()

    # Create visualizations
    print("\nGenerating visualizations...")

    # Static matplotlib visualization
    fig = visualizer.plot_hemisphere_heatmap(
        show_cameras=True,
        show_gradient=True,
        colormap='hot',
        save_path="C:\\Users\\rkhan\\Downloads\\file\hemisphere_analysis.png"
    )

    # Export results
    results_df = visualizer.export_results("C:\\Users\\rkhan\\Downloads\\file\\camera_analysis_results.csv")

    # Try to create interactive plot
    print("\nAttempting to create interactive plot...")
    try:
        visualizer.create_interactive_plot()
    except Exception as e:
        print(f"Could not create interactive plot: {e}")

    print("\n" + "="*50)
    print("Analysis complete!")
    print("Files generated:")
    print("  - hemisphere_analysis.png: Static visualization")
    print("  - camera_analysis_results.csv: Detailed results")
    print("  - interactive_hemisphere.html: Interactive 3D plot (if plotly installed)")
    print("  - sample_camera_data.csv: Sample input data")
