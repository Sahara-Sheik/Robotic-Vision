import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from hemisphere_with_robot import add_robot_to_hemisphere_3d


class HemisphereHeatmap:
    def __init__(self, n_cameras=100, radius=1.2):
        """
        Initialize the hemisphere heatmap visualizer.

        Args:
            n_cameras: Number of camera positions
            radius: Radius of the hemisphere (default 1.2 meters - physical size)
        """
        self.n_cameras = n_cameras
        self.radius = radius
        self.camera_positions = None
        self.accuracy_values = None

    def generate_camera_positions(self, distribution='uniform'):
        """
        Generate camera positions on a hemisphere.

        Args:
            distribution: 'uniform', 'random', or 'clustered'
        """
        if distribution == 'uniform':
            # Generate uniform distribution using Fibonacci spiral
            indices = np.arange(0, self.n_cameras, dtype=float) + 0.5
            theta = np.arccos(1 - indices/self.n_cameras)  # Polar angle (0 to pi/2 for hemisphere)
            phi = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden angle)

        elif distribution == 'random':
            # Random distribution
            phi = np.random.uniform(0, 2*np.pi, self.n_cameras)
            u = np.random.uniform(0, 1, self.n_cameras)
            theta = np.arccos(1 - u)  # For hemisphere

        else:  # clustered
            # Create clusters of cameras
            n_clusters = 5
            cluster_centers = np.random.uniform(0, 2*np.pi, n_clusters)
            phi = []
            theta = []
            for center in cluster_centers:
                n_in_cluster = self.n_cameras // n_clusters
                phi.extend(np.random.normal(center, 0.3, n_in_cluster))
                theta.extend(np.random.uniform(0, np.pi/2, n_in_cluster))
            phi = np.array(phi[:self.n_cameras])
            theta = np.array(theta[:self.n_cameras])

        # Convert spherical to Cartesian coordinates
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        self.camera_positions = np.column_stack([x, y, z])
        return self.camera_positions

    def generate_accuracy_values(self, pattern='gradient'):
        """
        Generate synthetic accuracy values for each camera position.

        Args:
            pattern: 'gradient', 'hotspots', 'radial', or 'sinusoidal'
        """
        if self.camera_positions is None:
            self.generate_camera_positions()

        x, y, z = self.camera_positions.T

        if pattern == 'gradient':
            # Linear gradient from one side to another
            self.accuracy_values = 0.5 + 0.5 * (x + y) / (2 * self.radius)

        elif pattern == 'hotspots':
            # Multiple hotspots of high accuracy
            hotspots = np.array([[0.5, 0.5, 0.7], [-0.5, 0.5, 0.7], [0, -0.7, 0.5]])
            accuracy = np.zeros(self.n_cameras)
            for hotspot in hotspots:
                dist = np.sqrt(np.sum((self.camera_positions - hotspot)**2, axis=1))
                accuracy += np.exp(-2 * dist**2)
            self.accuracy_values = accuracy / np.max(accuracy)

        elif pattern == 'radial':
            # Radial pattern from center
            r = np.sqrt(x**2 + y**2)
            self.accuracy_values = np.exp(-2 * r**2 / self.radius**2)

        else:  # sinusoidal
            # Sinusoidal pattern
            self.accuracy_values = 0.5 + 0.5 * np.sin(3*x) * np.cos(3*y)

        # Add some noise for realism
        self.accuracy_values += np.random.normal(0, 0.05, self.n_cameras)
        self.accuracy_values = np.clip(self.accuracy_values, 0, 1)

        return self.accuracy_values

    def create_hemisphere_mesh(self, resolution=50):
        """
        Create a hemisphere mesh for visualization.

        Args:
            resolution: Grid resolution for the mesh
        """
        theta = np.linspace(0, np.pi/2, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        THETA, PHI = np.meshgrid(theta, phi)

        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI)
        Z = self.radius * np.cos(THETA)

        return X, Y, Z, THETA, PHI

    def interpolate_accuracy(self, X, Y, Z):
        """
        Interpolate accuracy values across the hemisphere surface.

        Args:
            X, Y, Z: Mesh grid coordinates
        """
        # Flatten mesh points
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

        # Use RBF interpolation for smooth results
        rbf = RBFInterpolator(self.camera_positions, self.accuracy_values,
                             smoothing=0.1, kernel='gaussian', epsilon=2.0)
        interpolated = rbf(points)

        # Reshape to match mesh
        return interpolated.reshape(X.shape)

    def compute_gradient(self, accuracy_mesh, X, Y, Z):
        """
        Compute the gradient of accuracy across the hemisphere.

        Args:
            accuracy_mesh: Interpolated accuracy values
            X, Y, Z: Mesh coordinates
        """
        # Compute gradients in mesh space
        dy, dx = np.gradient(accuracy_mesh)

        # Scale gradients appropriately
        scale = 0.1
        dx *= scale
        dy *= scale

        # Project gradients to 3D surface
        # This is a simplified projection - for accurate surface gradients,
        # we would need to compute tangent vectors
        return dx, dy

    def plot_hemisphere_heatmap(self, show_cameras=True, show_gradient=True,
                                colormap='hot', save_path=None):
        """
        Create the complete visualization with heatmap and optional gradient arrows.

        Args:
            show_cameras: Whether to show camera positions
            show_gradient: Whether to show gradient arrows
            colormap: Matplotlib colormap name
            save_path: Path to save the figure
        """
        # Create mesh
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)

        # Interpolate accuracy
        accuracy_mesh = self.interpolate_accuracy(X, Y, Z)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8))
        # fig.patch.set_facecolor("#101014")


        # 3D Hemisphere view
        ax1 = fig.add_subplot(131, projection='3d')        #


        ## Remove 3D box/grid for ax1 but keep the hemisphere mesh
        ax1.grid(False)
        for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0  # hide lattice lines




        # Apply Gaussian filter for smoother visualization
        accuracy_smooth = gaussian_filter(accuracy_mesh, sigma=1)

        # Plot hemisphere with heatmap
        # --- Plot hemisphere with heatmap ---
        surf = ax1.plot_surface(
            X, Y, Z,
            facecolors=cm.get_cmap(colormap)(accuracy_smooth),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6   # make surface semi-transparent
        )

        # --- Add camera positions more visibly ---
        if show_cameras and self.camera_positions is not None:
            cam = self.camera_positions.copy()
            acc = getattr(self, "accuracy_values", np.ones(len(cam)))

            # normalize+offset EVEN IF caller forgot
            cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)

            ax1.scatter(
                cam[:, 0], cam[:, 1], cam[:, 2],
                s=10,                      # bigger
                c=acc, cmap=colormap,
                edgecolors='black',
                linewidths=0.9,
                depthshade=False,          # don't let 3d shading dim them
                alpha=1.0,
                zorder=10                  # draw on top
            )
        # if show_cameras and self.camera_positions is not None:
        #     cam = self.camera_positions.copy()
        #     acc = getattr(self, "accuracy_values", np.ones(len(cam)))

        #     # Slightly push cameras outward so they sit above the surface
        #     cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1.02)

        #     ax1.scatter(
        #         cam[:, 0], cam[:, 1], cam[:, 2],
        #         s=70,                        # bigger points
        #         c=acc, cmap=colormap,
        #         edgecolors='black',          # dark outline for visibility
        #         linewidths=0.8,
        #         depthshade=False,            # ensures bright color
        #         alpha=1.0,
        #         zorder=5
        #     )
        # Add gradient arrows
        if show_gradient:
            dx, dy = self.compute_gradient(accuracy_smooth, X, Y, Z)

            # Sample points for arrows (every nth point)
            step = 5
            X_sample = X[::step, ::step]
            Y_sample = Y[::step, ::step]
            Z_sample = Z[::step, ::step]
            dx_sample = dx[::step, ::step]
            dy_sample = dy[::step, ::step]

            # Create tangent vectors for arrows
            for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    if Z_sample[i, j] > 0.05:  # Only show arrows on visible part
                        # Compute approximate tangent vectors
                        start_point = [X_sample[i, j], Y_sample[i, j], Z_sample[i, j]]

                        # Gradient direction in 3D (simplified)
                        grad_x = dx_sample[i, j] * np.cos(PHI[i*step, j*step])
                        grad_y = dy_sample[i, j] * np.sin(PHI[i*step, j*step])
                        grad_z = 0

                        ax1.quiver(start_point[0], start_point[1], start_point[2],
                                 grad_x, grad_y, grad_z,
                                 color='black', alpha=0.5, arrow_length_ratio=0.3,
                                 length=0.05)

        ax1.set_xlabel('X (m)', fontsize=9)
        ax1.set_ylabel('Y (m)', fontsize=9)
        ax1.set_zlabel('Z (m)', fontsize=9)
        # Title removed
        ax1.set_box_aspect([1,1,0.5])
        ax1.view_init(elev=30, azim=45)
        ax1.tick_params(labelsize=8)

        # Top-down view (circular heatmap)
        ax2 = fig.add_subplot(132)

        # Project to 2D circle
        r = np.sqrt(X**2 + Y**2)
        mask = r <= self.radius

        ax2 = fig.add_subplot(132)
        # ax2.set_facecolor("#101014")

        # Create circular heatmap
        contourf = ax2.contourf(X, Y, accuracy_smooth, levels=20, cmap=colormap)
        ax2.contour(X, Y, accuracy_smooth, levels=10, colors='black',
                   linewidths=0.5, alpha=0.3)

        if show_cameras:
            ax2.scatter(self.camera_positions[:, 0],
                       self.camera_positions[:, 1],
                       c=self.accuracy_values, cmap=colormap,
                       s=30, edgecolors='black', linewidths=0.5)

        # Add gradient arrows in 2D
        if show_gradient:
            ax2.quiver(X[::step, ::step], Y[::step, ::step],
                      dx[::step, ::step], dy[::step, ::step],
                      alpha=0.5, width=0.003)

        ax2.set_xlabel('X (m)', fontsize=9)
        ax2.set_ylabel('Y (m)', fontsize=9)
        # Title removed
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)
        # ax2.set_xlim([-self.radius*1.1, self.radius*1.1])
        # ax2.set_ylim([-self.radius*1.1, self.radius*1.1])
        ax2.set_xlim([-self.radius*1, self.radius*1])
        ax2.set_ylim([-self.radius*1, self.radius*1])

        # Add circle boundary
        # circle = plt.Circle((0, 0), self.radius, fill=False, edgecolor='black', linewidth=2)
        # ax2.add_patch(circle)
        # thick faint ring
        # ax2.add_patch(plt.Circle((0,0), 1.0, fill=False, edgecolor="white", linewidth=2.0, alpha=0.25, zorder=1))
        # # sharp dark ring on top
        # ax2.add_patch(plt.Circle((0,0), 1.0, fill=False, edgecolor="black", linewidth=0.8, zorder=30))


        # Side view with gradient magnitude

        ax3 = fig.add_subplot(133, projection='3d')

        # Remove 3D box/grid for ax3 but keep the hemisphere mesh
        ax3.grid(False)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0


        # Compute gradient magnitude
        grad_magnitude = np.sqrt(dx**2 + dy**2)

        # --- Plot hemisphere surface (slightly transparent) ---
        surf2 = ax3.plot_surface(
            X, Y, Z,
            facecolors=cm.get_cmap('hot')(grad_magnitude),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6
        )

            # normalize+offset


        # --- Add camera positions ---
        if show_cameras and self.camera_positions is not None:
            cam = self.camera_positions.copy()
            acc = getattr(self, "accuracy_values", np.ones(len(cam)))

            # Push cameras outward slightly from the hemisphere
            cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)

            # ax3.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=70,
            #     c=acc, cmap='turbo',         # you can change to any color map (e.g., 'hot' or 'inferno')
            #     edgecolors='black',
            #     linewidths=0.8,
            #     depthshade=False,
            #     alpha=1.0,
            #     zorder=5
            # )
            # ax3.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=10,                      # bigger
            #     c=acc, cmap='hot',
            #     edgecolors='black',
            #     linewidths=0.9,
            #     depthshade=False,          # don't let 3d shading dim them
            #     alpha=1.0,
            #     zorder=10                  # draw on top
            # )
        ax3.set_xlabel('X (m)', fontsize=9)
        ax3.set_ylabel('Y (m)', fontsize=9)
        ax3.set_zlabel('Z (m)', fontsize=9)
        # Title removed
        ax3.set_box_aspect([1,1,0.5])
        ax3.view_init(elev=30, azim=135)
        ax3.tick_params(labelsize=8)

        # ax3 = fig.add_subplot(133, projection='3d')

        # # Compute gradient magnitude
        # grad_magnitude = np.sqrt(dx**2 + dy**2)

        # # Plot with gradient magnitude as color
        # surf2 = ax3.plot_surface(X, Y, Z, facecolors=cm.get_cmap('viridis')(grad_magnitude),
        #                         alpha=0.9, shade=True, antialiased=True)

        # ax3.set_xlabel('X')
        # ax3.set_ylabel('Y')
        # ax3.set_zlabel('Z')
        # # ax3.set_title('Gradient Magnitude Visualization')  # Title removed
        # ax3.set_box_aspect([1,1,0.5])
        # ax3.view_init(elev=30, azim=135)

        # Add colorbars
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95, wspace=0.3)

        # Accuracy colorbar
        cbar_ax1 = fig.add_axes([0.12, 0.05, 0.25, 0.02])
        cbar1 = plt.colorbar(contourf, cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label('Accuracy', fontsize=10)

        # Gradient magnitude colorbar
        cbar_ax2 = fig.add_axes([0.68, 0.05, 0.25, 0.02])
        norm = Normalize(vmin=grad_magnitude.min(), vmax=grad_magnitude.max())
        sm = cm.ScalarMappable(norm=norm, cmap='hot')
        cbar2 = plt.colorbar(sm, cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label('Gradient Magnitude', fontsize=10)

        plt.suptitle('Camera Position Accuracy Analysis on Hemisphere', fontsize=16, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # plt.show()

        return fig


    def plot_paper_topdown(
        self,
        save_path,
        fig_width_in=3.25,
        dpi=300,
        cmap="hot",
        label=None,     # e.g. "height"
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        if self.camera_positions is None or self.accuracy_values is None:
            raise ValueError("Need camera_positions + accuracy_values")

        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=90)
        acc_mesh = self.interpolate_accuracy(X, Y, Z)

        dx_raw, dy_raw = self.compute_gradient(acc_mesh, X, Y, Z)
        grad_mag = np.sqrt(dx_raw**2 + dy_raw**2)
        max_mag = np.max(grad_mag)
        if max_mag < 1e-12:
            dx = np.zeros_like(dx_raw)
            dy = np.zeros_like(dy_raw)
        else:
            dx = dx_raw / (max_mag + 1e-12)
            dy = dy_raw / (max_mag + 1e-12)

        fig, ax = plt.subplots(figsize=(fig_width_in, fig_width_in), dpi=dpi)


        cf = ax.contourf(X, Y, acc_mesh, levels=30, cmap=cmap)

        # ax.scatter(
        #     # self.camera_positions[:, 0],
        #     # self.camera_positions[:, 1],
        #     c=self.accuracy_values,
        #     cmap=cmap,
        #     s=18,
        #     # edgecolors="black",
        #     linewidths=0.4,
        #     zorder=10,
        # )

        # step = 5
        # ax.quiver(
        #     X[::step, ::step],
        #     Y[::step, ::step],
        #     dx[::step, ::step],
        #     dy[::step, ::step],
        #     color="black",
        #     width=0.0045,
        #     headwidth=6.0,
        #     headlength=6.0,
        #     scale=18,
        #     alpha=0.85,
        #     zorder=20,
        # )

        step = 8
        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            dx[::step, ::step],
            dy[::step, ::step],
            color="white",
            width=0.005,
            headwidth=6.5,
            headlength=7.0,
            scale=16,
            alpha=0.8,
            zorder=50,
        )


        circ = plt.Circle((0, 0), self.radius, fill=False, edgecolor="black", linewidth=1.1, zorder=25)
        ax.add_patch(circ)

        ax.set_aspect("equal")
        ax.set_xlim(-self.radius*1.05, self.radius*1.05)
        ax.set_ylim(-self.radius*1.05, self.radius*1.05)
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.tick_params(labelsize=6)

        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Accuracy", fontsize=7)

        # Caption at bottom - REMOVED to keep images clean
        # if label is not None:
        #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

        fig.tight_layout(pad=0.4)
        fig.savefig(save_path, dpi=dpi)
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"))
        plt.close(fig)

    def plot_paper_3d(
        self,
        save_path,
        fig_width_in=3.25,
        dpi=300,
        cmap="hot",
        label=None,
        robot_image_path=None,
        size=0.03,

    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        import numpy as np

        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)
        acc_mesh = self.interpolate_accuracy(X, Y, Z)

        fig = plt.figure(figsize=(fig_width_in, fig_width_in * 0.85), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
        colors = plt.cm.hot(acc_mesh)  # (n,m,4)
        colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])  # brighten
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0


        ax.plot_surface(
            X, Y, Z,
            facecolors=colors,
            # facecolors=plt.cm.get_cmap(cmap)(acc_mesh),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False, alpha=0.6,
        )

        cam = self.camera_positions.copy()
        # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)
        # ax.scatter(
        #     cam[:, 0], cam[:, 1], cam[:, 2],
        #     s=10,
        #     c=self.accuracy_values,
        #     cmap=cmap,
        #     edgecolors="black",
        #     linewidths=0.4,
        #     depthshade=False,
        #     zorder=5,
        # )

        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_zlabel("Z (m)", fontsize=7, labelpad=4)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect([1, 1, 0.5])
        ax.view_init(elev=28, azim=40)

        # move Z label to the left so it doesn't sit on the colorbar
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.set_label_coords(-0.08, 0.5)   # (x, y) in axes coords

        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(acc_mesh)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.06,          # was 0.03 → push to the right
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Accuracy", fontsize=7)

        # Bottom label removed to keep images clean
        # if label is not None:
        #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

        # Add robot image at the center, if provided
        if robot_image_path:
            try:
                add_robot_to_hemisphere_3d(
                    ax,
                    robot_image_path,
                    position=(-0.3, 0, 0.1*self.radius),   # scale with radius
                    size=0.030                           # proper proportional size
                )
            except Exception as e:
                print(f"Could not add robot image in plot_paper_3d: {e}")


        fig.tight_layout(pad=0.4)
        fig.savefig(save_path, dpi=dpi)
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"))
        plt.close(fig)


    def plot_paper_gradient3d(
        self,
        save_path,
        fig_width_in=3.25,
        dpi=300,
        surf_cmap="hot",
        cam_cmap="hot",
        label=None,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        if self.camera_positions is None or self.accuracy_values is None:
            raise ValueError("Need camera_positions + accuracy_values")
        # remove the XYZ box / grid, keep only the hemisphere

        # hide pane backgrounds and their grid lines
        # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        #     axis.pane.set_visible(False)
        #     axis._axinfo["grid"]["linewidth"] = 0


        # 1) mesh + gradient
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)
        acc_mesh = self.interpolate_accuracy(X, Y, Z)
        dx, dy = self.compute_gradient(acc_mesh, X, Y, Z)
        grad_mag = np.sqrt(dx**2 + dy**2)

        # 2) figure
        fig = plt.figure(figsize=(fig_width_in, fig_width_in * 0.85), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0


        light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
        colors = plt.cm.hot(acc_mesh)  # (n,m,4)
        colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])  # brighten
        # 3) surface

        ax.plot_surface(
            X, Y, Z,
            # facecolors=plt.cm.get_cmap(surf_cmap)(grad_mag),
            facecolors = colors,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6,
            zorder=1,
        )

        # # 4) cameras (pushed out)
        # cam = self.camera_positions.copy()
        # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)
        # ax.scatter(
        #     cam[:, 0], cam[:, 1], cam[:, 2],
        #     s=10,
        #     c=self.accuracy_values,
        #     cmap=cam_cmap,
        #     edgecolors="black",
        #     linewidths=0.4,
        #     depthshade=False,
        #     zorder=5,
        # )

        # basic axes
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect([1, 1, 0.5])
        ax.view_init(elev=28, azim=135)

        # 5) HARD-HIDE the original Z axis so it can’t appear on the right
        ax.set_zlabel("")                    # blank
        try:
            ax.zaxis.label.set_visible(False)
        except Exception:
            pass
        # hide z ticks & tick labels
        try:
            for item in ax.zaxis.get_ticklines() + ax.zaxis.get_ticklabels():
                item.set_visible(False)
        except Exception:
            pass
        # in some mpl versions:
        if hasattr(ax, "w_zaxis"):
            ax.w_zaxis.line.set_lw(0.)
            ax.w_zaxis.set_pane_color((0, 0, 0, 0))

        # 6) colorbar on the right
        mappable = plt.cm.ScalarMappable(cmap=surf_cmap)
        mappable.set_array(grad_mag)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.06,   # push out a bit
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Gradient Magnitude", fontsize=7)

        # 7) now draw OUR Z on the left of the figure -> this will not move
        fig.text(
            0.045, 0.55,  # left margin
            "Z (m)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
        )

        # Bottom label removed to keep images clean
        # if label is not None:
        #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

        fig.tight_layout(pad=0.4)
        fig.savefig(save_path, dpi=dpi)
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"))
        plt.close(fig)


    @staticmethod

    def make_cvpr_3x2_spheres(components_data, output_path, cmap="hot"):
        """
        CVPR-style 3x2 grid, but each panel is the 3D hemisphere surface.
        components_data: list of dicts:
            {
              "name": "height",
              "positions": (N,3) np.array,
              "accuracy":  (N,)   np.array
            }
        Uses first 6 entries.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # 2-column CVPR width
        fig_w = 6.75
        fig_h = 7.0
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

        # only 6
        components_data = components_data[:6]

        # to get global min/max for shared colorbar
        all_vals = []

        for idx, comp in enumerate(components_data):
            # 3 rows x 2 cols
            ax = fig.add_subplot(3, 2, idx + 1, projection="3d")
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["linewidth"] = 0


            vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
            vis.camera_positions = comp["positions"]
            vis.accuracy_values = comp["accuracy"]

            X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=45)
            acc_mesh = vis.interpolate_accuracy(X, Y, Z)
            all_vals.append(acc_mesh)

            # Scale to 1.2m physical size for display
            X = X * 1.2
            Y = Y * 1.2
            Z = Z * 1.2

            light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
            colors = plt.cm.hot(acc_mesh)  # (n,m,4)
            colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])  # brighten

            ax.plot_surface(
                X, Y, Z,
                # facecolors=plt.cm.get_cmap(cmap)(acc_mesh),
                facecolors = colors,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False,
                shade=False,
                alpha=0.65,
            )

            # ADD ROBOT IMAGE AT CENTER
            # try:
            #     from PIL import Image
            #     import matplotlib.patches as mpatches
            #     from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            #     # Load and rotate robot image
            #     robot_img = Image.open("C:\\Users\\rkhan\\Downloads\\robot.png")
            #     robot_img = robot_img.rotate(90, expand=True)  # Counter-clockwise

            #     if robot_img.mode != 'RGBA':
            #         robot_img = robot_img.convert('RGBA')

            #     # Make semi-transparent
            #     img_array = np.array(robot_img)
            #     if img_array.shape[2] == 4:
            #         img_array[:, :, 3] = (img_array[:, :, 3] * 0.6).astype(np.uint8)
            #     robot_img_transparent = Image.fromarray(img_array)

            try:
                ROBOT_IMAGE_PATH = "C:\\Users\\rkhan\\Downloads\\robot.png"
                add_robot_to_hemisphere_3d(
                    ax,
                    ROBOT_IMAGE_PATH,
                    position=(-0.1, 0.0, 0.15*1.2),   # scale with 1.2m radius
                    size=0.030                         # proper proportional size (40%)
                )
            except Exception as e:
                print(f"Could not add robot to panel {comp['name']}: {e}")

            #     # Create a vertical plane for robot at center
            #     robot_size = 0.3
            #     vertices = [
            #         [-robot_size/2, 0, 0],           # Bottom left
            #         [robot_size/2, 0, 0],            # Bottom right
            #         [robot_size/2, 0, robot_size],   # Top right
            #         [-robot_size/2, 0, robot_size]   # Top left
            #     ]

            #     # Add as a semi-transparent plane
            #     poly = Poly3DCollection([vertices], alpha=0.7, facecolor='lightgray',
            #                            edgecolor='white', linewidth=0.5, zorder=100)
            #     ax.add_collection3d(poly)

            #     # Add text indicator
            #     ax.text(0, 0, 0.1, '🤖', fontsize=20, ha='center', va='center',
            #            zorder=200, bbox=dict(boxstyle='round', facecolor='white',
            #            alpha=0.6, pad=0.2, edgecolor='none'))

            # except Exception as e:
            #     # Fallback: just add emoji
            #     ax.text(0, 0, 0.1, '🤖', fontsize=20, ha='center', va='center',
            #            zorder=200, bbox=dict(boxstyle='round', facecolor='white',
            #            alpha=0.6, pad=0.2, edgecolor='none'))

            # cameras → push out
            # cam = vis.camera_positions.copy()
            # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (vis.radius * 1)
            # ax.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=10,
            #     c=vis.accuracy_values,
            #     cmap=cmap,
            #     edgecolors="black",
            #     linewidths=0.35,
            #     depthshade=False,
            #     zorder=5,
            # )

            ax.set_xlabel("X (m)", fontsize=6)
            ax.set_ylabel("Y (m)", fontsize=6)
            ax.set_zlabel("Z (m)", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.set_box_aspect([1, 1, 0.5])
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_zlim([0, 1.3])
            # alternate views a bit so they don’t self-hide
            if idx % 2 == 0:
                ax.view_init(elev=26, azim=38)
            else:
                ax.view_init(elev=26, azim=135)

            # Label removed to keep images clean
            # ax.text2D(
            #     0.5, -0.18,
            #     comp["name"].replace("_", " "),
            #     transform=ax.transAxes,
            #     ha="center",
            #     va="top",
            #     fontsize=6.2,
            # )

        # shared colorbar
        all_vals = np.array(all_vals)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())

        cax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("Accuracy", fontsize=7)

        fig.subplots_adjust(
            left=0.03, right=0.90, top=0.97, bottom=0.05,
            wspace=0.05, hspace=0.25
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def make_cvpr_2x3_grid(components_data, output_path, cam_cmap="hot"):
        """
        CVPR-style 3x2 figure (3 rows, 2 cols) for all components.

        components_data: list of dicts like
            {
            "name": "height",
            "positions": np.array shape (N,3),
            "accuracy": np.array shape (N,)
            }
        Only the first 6 items are used.
        """
        # CVPR 2-column width ≈ 6.75 in
        fig_w = 6.75
        fig_h = 7.0   # taller than 2x3 to fit 3 rows

        # make 3x2 axes
        fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h), dpi=300)
        axes = axes.reshape(3, 2)
        # fig.patch.set_facecolor("#101014")
        # for ax in axes.ravel():
        #     ax.set_facecolor("#101014")

        all_mesh_vals = []

        from hemisphere_heatmap import HemisphereHeatmap  # if this is the same file, it's fine

        # we only need 6
        components_data = components_data[:6]

        for idx, comp in enumerate(components_data):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
            vis.camera_positions = comp["positions"]
            vis.accuracy_values = comp["accuracy"]

            # mesh + interp
            X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=70)
            acc_mesh = vis.interpolate_accuracy(X, Y, Z)
            all_mesh_vals.append(acc_mesh)

            # heatmap
            ax.contourf(X, Y, acc_mesh, levels=25, cmap=cam_cmap)

            # # cameras
            # ax.scatter(
            #     vis.camera_positions[:, 0],
            #     vis.camera_positions[:, 1],
            #     c=vis.accuracy_values,
            #     cmap=cam_cmap,
            #     s=10,
            #     edgecolors="black",
            #     linewidths=0.25,
            #     zorder=5,
            # )

            # circle outline
            # thick faint ring
            ax.add_patch(plt.Circle((0,0), 1.2, fill=False, edgecolor="white", linewidth=2.0, alpha=0.25, zorder=1))
            # sharp dark ring on top
            ax.add_patch(plt.Circle((0,0), 1.2, fill=False, edgecolor="black", linewidth=0.8, zorder=30))

            # circ = plt.Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=0.7)
            # ax.add_patch(circ)

            # tidy axes
            ax.set_aspect("equal")
            # ax.set_xlim(-1.05, 1.05)
            # ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(-1.2*1.05, 1.2*1.05)
            ax.set_ylim(-1.2*1.05, 1.2*1.05)

            ax.set_xticks([])
            ax.set_yticks([])

            # component label under panel
            ax.text(
                0.5, -0.10,
                comp["name"].replace("_", " "),
                ha="center", va="top",
                transform=ax.transAxes,
                fontsize=6.5,
            )

        # shared colorbar on right
        all_mesh_vals = np.array(all_mesh_vals)
        vmin = float(all_mesh_vals.min())
        vmax = float(all_mesh_vals.max())

        # [left, bottom, width, height]
        cax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
        sm = plt.cm.ScalarMappable(cmap=cam_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("Accuracy", fontsize=7)

        fig.subplots_adjust(
            left=0.05,
            right=0.90,
            top=0.97,
            bottom=0.05,
            wspace=0.08,
            hspace=0.20,
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    def generate_report(self):
        """Generate a statistical report of the accuracy distribution."""
        if self.accuracy_values is None:
            print("No accuracy values generated yet!")
            return

        print("\n" + "="*50)
        print("ACCURACY ANALYSIS REPORT")
        print("="*50)
        print(f"Number of camera positions: {self.n_cameras}")
        print(f"Hemisphere radius: {self.radius}")
        print(f"\nAccuracy Statistics:")
        print(f"  Mean accuracy: {np.mean(self.accuracy_values):.3f}")
        print(f"  Std deviation: {np.std(self.accuracy_values):.3f}")
        print(f"  Min accuracy: {np.min(self.accuracy_values):.3f}")
        print(f"  Max accuracy: {np.max(self.accuracy_values):.3f}")

        # Find best camera position
        best_idx = np.argmax(self.accuracy_values)
        best_pos = self.camera_positions[best_idx]
        print(f"\nBest camera position:")
        print(f"  Index: {best_idx}")
        print(f"  Position: ({best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f})")
        print(f"  Accuracy: {self.accuracy_values[best_idx]:.3f}")

        # Find worst camera position
        worst_idx = np.argmin(self.accuracy_values)
        worst_pos = self.camera_positions[worst_idx]
        print(f"\nWorst camera position:")
        print(f"  Index: {worst_idx}")
        print(f"  Position: ({worst_pos[0]:.3f}, {worst_pos[1]:.3f}, {worst_pos[2]:.3f})")
        print(f"  Accuracy: {self.accuracy_values[worst_idx]:.3f}")
        print("="*50 + "\n")


# Main execution
if __name__ == "__main__":
    # Create visualizer
    visualizer = HemisphereHeatmap(n_cameras=100)

    # Generate camera positions (try 'uniform', 'random', or 'clustered')
    visualizer.generate_camera_positions(distribution='uniform')

    # Generate accuracy values (try 'gradient', 'hotspots', 'radial', or 'sinusoidal')
    visualizer.generate_accuracy_values(pattern='hotspots')

    # Generate report
    visualizer.generate_report()

    # Create visualization
    fig = visualizer.plot_hemisphere_heatmap(
        show_cameras=False,
        show_gradient=True,
        colormap='hot',  # Try also: 'viridis', 'plasma', 'coolwarm', 'seismic'
        save_path='C;\\Users\\rkhan\\Downloads\\hemisphere_heatmap.png'
    )

    print("Visualization saved to hemisphere_heatmap.png")
