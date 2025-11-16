"""
Modified Hemisphere Heatmap with Robot Image in Center
Adds robot image placement functionality to single hemisphere visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import MaxNLocator

def add_robot_to_hemisphere_3d(ax, robot_image_path, position=(-0.4, 0, 0),
                                size=0.10, rotation_angle=0):
    """
    Add a robot image to the center of a 3D hemisphere plot.

    Args:
        ax: 3D axes object
        robot_image_path: Path to robot image file
        position: (x, y, z) tuple for robot position (default center at origin)
        size: Size of the robot image as fraction of hemisphere (default 0.40 = 40%)
        rotation_angle: Rotation angle in degrees
    """

# def add_robot_image_3d(ax, robot_image_path, position=(0, 0, 0), size=0.3, rotation_angle=0):
#     """
#     ax         : 3D Axes (projection='3d')
#     image_path : path to robot PNG (ideally with transparent background)
#     xyz        : 3D coordinates where the robot should sit
#     zoom       : scale factor for the image
#     """
    img = mpimg.imread(robot_image_path)
    from scipy import ndimage
    # img = ndimage.rotate(img, 135, reshape=True, order=1)
    imagebox = OffsetImage(img, zoom=size)

    x3d, y3d, z3d = position
    x2d, y2d, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())

    ab = AnnotationBbox(
        imagebox,
        (x2d, y2d),
        xycoords='data',
        frameon=False
    )
    ax.add_artist(ab)
    return ab

    # try:
    #     # Load robot image
    #     robot_img = Image.open(robot_image_path)

    #     # Convert to RGBA if needed
    #     if robot_img.mode != 'RGBA':
    #         robot_img = robot_img.convert('RGBA')

    #     # Make background transparent (if white)
    #     img_array = np.array(robot_img)

    #     # Create a vertical plane at the center with the robot image
    #     # We'll use a simple approach: create a textured plane

    #     # Define plane vertices (vertical plane at center)
    #     x0, y0, z0 = position

    #     # Create vertices for a vertical rectangle
    #     vertices = [
    #         [x0 - size/2, y0, z0],           # Bottom left
    #         [x0 + size/2, y0, z0],           # Bottom right
    #         [x0 + size/2, y0, z0 + size],    # Top right
    #         [x0 - size/2, y0, z0 + size]     # Top left
    #     ]

    #     # Create the polygon collection
    #     poly = Poly3DCollection([vertices], alpha=0.9)

    #     # For now, we'll just add a colored box as placeholder
    #     # True image texturing in matplotlib 3D is very limited
    #     poly.set_facecolor('lightgray')
    #     poly.set_edgecolor('black')
    #     ax.add_collection3d(poly)

    #     # Add a text label
    #     ax.text(x0, y0, z0 + size/2, '🤖', fontsize=60,
    #            ha='center', va='center', zorder=100,
    #            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #     return True

    # except Exception as e:
    #     print(f"Could not load robot image: {e}")
    #     # Add simple robot representation
    #     ax.text(position[0], position[1], position[2], '🤖',
    #            fontsize=60, ha='center', va='center', zorder=100)
    #     return False


def plot_hemisphere_with_robot_topdown(vis, output_path, robot_image_path=None,
                                       cmap='hot_r', robot_size=0.40):
    """
    Create a top-down view of hemisphere with robot image in center.
    This is the most effective way to show robot in the visualization.

    Args:
        vis: HemisphereHeatmap object with camera_positions and accuracy_values
        output_path: Path to save the figure
        robot_image_path: Path to robot image file
        cmap: Colormap name
        robot_size: Size of robot image as fraction of hemisphere radius
                   Default 0.40 represents 19" robot (0.48m) in 1.2m hemisphere
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)

    # Create hemisphere mesh
    X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=100)

    # Interpolate accuracy
    mse_mesh = vis.interpolate_mse(X, Y, Z)

    # Create heatmap
    from scipy.ndimage import gaussian_filter
    mse_smooth = gaussian_filter(mse_mesh, sigma=1)

    # Plot contour (X, Y already scaled to vis.radius which is 1.2m)
    contourf = ax.contourf(X, Y, mse_smooth, levels=25, cmap=cmap)

    # Add gradient arrows to show direction of improvement
    dx_raw, dy_raw = vis.compute_gradient(mse_smooth, X, Y, Z)
    grad_mag = np.sqrt(dx_raw**2 + dy_raw**2)
    max_mag = np.max(grad_mag)

    if max_mag > 1e-12:
        # Normalize gradient direction
        ux = dx_raw / (max_mag + 1e-12)
        uy = dy_raw / (max_mag + 1e-12)

        # Normalize magnitudes to [0, 1]
        grad_norm = grad_mag / max_mag

        # Maximum arrow length in hemisphere units
        L_max = 0.5 * vis.radius

        # Final arrow components
        dx = ux * grad_norm * L_max
        dy = uy * grad_norm * L_max

        # Subsample arrows so it's not too busy
        step = 8
        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            dx[::step, ::step],
            dy[::step, ::step],
            color="tab:blue",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.008,      # Increased from 0.004 - makes shaft thicker
            headwidth=7,      # Increased from 5 - makes head wider
            headlength=8,     # Increased from 6 - makes head longer
            alpha=0.9,
            zorder=5,  # Below robot (which is zorder=10)
        )

    # Add circle boundary
    circle = plt.Circle((0, 0), vis.radius, fill=False,
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)

    # Add robot image in center if provided
    if robot_image_path:
        try:
            robot_img = Image.open(robot_image_path)

            # Rotate 90 degrees counter-clockwise
            robot_img = robot_img.rotate(90, expand=True)

            # Convert to RGBA for transparency
            if robot_img.mode != 'RGBA':
                robot_img = robot_img.convert('RGBA')
            y_offset = 0.2 * vis.radius  # Adjust this value (0.2 = 20% of radius)

            # Calculate image extent (incase we want to change robot images robot_size is fraction of hemisphere radius )
            extent = [-robot_size*vis.radius, robot_size*vis.radius,
                     -robot_size*vis.radius+y_offset, robot_size*vis.radius+y_offset]

            # Display robot image at center with transparency
            ax.imshow(robot_img, extent=extent, zorder=10, alpha=0.6)

            # Add subtle border around robot (also transparent)
            # rect = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
            #                          robot_size*2*vis.radius, robot_size*2*vis.radius,
            #                          fill=False, edgecolor='white',
            #                          linewidth=2, zorder=11, alpha=0.7)
            # ax.add_patch(rect)

            # Add inner shadow effect for depth
            rect2 = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
                                      robot_size*2*vis.radius, robot_size*2*vis.radius,
                                      fill=False, edgecolor='black',
                                      linewidth=1, zorder=11, alpha=0.3)
            # ax.add_patch(rect2)

        except Exception as e:
            print(f"Could not load robot image: {e}")
            # Add text placeholder
            ax.text(0, 0, '🤖', fontsize=50, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                   zorder=10)

    # Configure axes (show actual 1.2m physical size)
    ax.set_aspect('equal')
    ax.set_xlim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_ylim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    # Title removed - will be added externally as needed
    ax.tick_params(labelsize=9)
    ax.set_axis_off()

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8,orientation='horizontal')
    cbar.set_label('MSE', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    # Save to temporary file first
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name

    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Aggressively crop white margins
    img = Image.open(temp_path).convert("RGB")
    img_array = np.array(img)

    # Very aggressive threshold - only skip nearly pure white
    threshold = 254

    # Check all three channels
    is_white = (img_array[:, :, 0] >= threshold) & \
               (img_array[:, :, 1] >= threshold) & \
               (img_array[:, :, 2] >= threshold)

    # Find rows and columns with non-white content
    non_white_rows = np.where(~is_white.all(axis=1))[0]
    non_white_cols = np.where(~is_white.all(axis=0))[0]

    if len(non_white_rows) > 0 and len(non_white_cols) > 0:
        # Get bounding box with minimal padding
        pad = 5  # Just 5 pixels to keep colorbar labels visible
        top = max(0, non_white_rows.min() - pad)
        bottom = min(img_array.shape[0], non_white_rows.max() + pad)
        left = max(0, non_white_cols.min() - pad)
        right = min(img_array.shape[1], non_white_cols.max() + pad)

        # Crop to tight bounding box
        img_cropped = img.crop((left, top, right, bottom))
    else:
        img_cropped = img

    # Clean up temp file
    os.unlink(temp_path)

    # Save final cropped image
    img_cropped.save(output_path)

    print(f"Saved hemisphere with robot to {output_path}")


def plot_hemisphere_with_robot_3d(vis, output_path, robot_image_path=None,
                                  cmap='hot_r', show_cameras=False):
    """
    Create a 3D hemisphere view with robot representation at center.

    Args:
        vis: HemisphereHeatmap object
        output_path: Path to save figure
        robot_image_path: Path to robot image
        cmap: Colormap name
        show_cameras: Whether to show camera positions
    """
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh
    X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=50)

    # Interpolate accuracy
    mse_mesh = vis.interpolate_mse(X, Y, Z)

    # Apply smoothing
    from scipy.ndimage import gaussian_filter
    mse_smooth = gaussian_filter(mse_mesh, sigma=1)

    # Plot hemisphere surface
    import matplotlib.cm as cm
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=cm.get_cmap(cmap)(mse_smooth),
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.7
    )

    # Add camera positions if requested
    if show_cameras and vis.camera_positions is not None:
        cam = vis.camera_positions.copy()
        acc = vis.accuracy_values

        # Normalize
        cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (vis.radius * 1.05)

        ax.scatter(
            cam[:, 0], cam[:, 1], cam[:, 2],
            s=30, c=acc, cmap=cmap,
            edgecolors='black', linewidths=0.5,
            depthshade=False, alpha=1.0,
            zorder=10
        )

    # Add robot at center
    if robot_image_path:
        add_robot_to_hemisphere_3d(ax, robot_image_path,
                                   position=(-0.4, 0, 0.1*vis.radius),
                                   size=0.02)
    else:
        # Add robot emoji as placeholder
        ax.text(0, 0, 0.1*vis.radius, '🤖', fontsize=70,
               ha='center', va='center', zorder=100,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Configure plot
    ax.grid(False)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_box_aspect([1, 1, 0.5])
    ax.view_init(elev=25, azim=45)
    ax.set_xlim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_ylim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_zlim([0, vis.radius*1.1])
    ax.tick_params(labelsize=9)

    # Add colorbar
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=mse_smooth.min(), vmax=mse_smooth.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(mse_smooth)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, orientation='horizontal')
    cbar.set_label('MSE', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Title removed

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D hemisphere with robot to {output_path}")


def plot_hemisphere_with_robot_combined(vis, output_path, robot_image_path_3d,robot_image_path_top,
                                        cmap='hot_r', show_cameras=False):
    """
    Create a combined view with both 3D and top-down perspectives.
    Robot image is shown in the top-down view.

    Args:
        vis: HemisphereHeatmap object
        output_path: Path to save figure
        robot_image_path: Path to robot image
        cmap: Colormap name
        show_cameras: Whether to show camera positions
    """
    fig = plt.figure(figsize=(16, 8), dpi=300)

    # Create mesh and interpolate
    X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=50)
    mse_mesh = vis.interpolate_mse(X, Y, Z)

    from scipy.ndimage import gaussian_filter
    mse_smooth = gaussian_filter(mse_mesh, sigma=1)

    # LEFT: 3D View
    ax1 = fig.add_subplot(121, projection='3d')

    import matplotlib.cm as cm
    surf = ax1.plot_surface(
        X, Y, Z,
        facecolors=cm.get_cmap(cmap)(mse_smooth),
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.7
    )

    if show_cameras and vis.camera_positions is not None:
        cam = vis.camera_positions.copy()
        cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (vis.radius * 1.05)
        ax1.scatter(cam[:, 0], cam[:, 1], cam[:, 2],
                   s=20, c=vis.mse_values, cmap=cmap ,
                   edgecolors='black', linewidths=0.5,
                   depthshade=False, zorder=10)

    # Add robot at center
    if robot_image_path_3d:
        add_robot_to_hemisphere_3d(ax1, robot_image_path_3d,
                                   position=(-0.3, 0.9, 0.1*vis.radius),
                                   size=0.08)
    else:
        # Add robot emoji as placeholder
        ax1.text(0, 0, 0.1*vis.radius, '🤖', fontsize=70,
               ha='center', va='center', zorder=100,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.grid(False)
    ax1.set_xlabel('X (m)', fontsize=9)
    ax1.set_ylabel('Y (m)', fontsize=9)
    ax1.set_zlabel('Z (m)', fontsize=9)
    ax1.set_box_aspect([1, 1, 0.5])
    ax1.view_init(elev=25, azim=45)
    ax1.set_xlim([-vis.radius*1.1, vis.radius*1.1])
    ax1.set_ylim([-vis.radius*1.1, vis.radius*1.1])
    ax1.set_zlim([0, vis.radius*1.1])
    ax1.tick_params(labelsize=8)
    # Title removed

    # RIGHT: Top-Down View with Robot Image
    ax2 = fig.add_subplot(122)

    contourf = ax2.contourf(X, Y, mse_smooth, levels=25, cmap=cmap)

    # Circle boundary
    circle = plt.Circle((0, 0), vis.radius, fill=False,
                       edgecolor='black', linewidth=2)
    ax2.add_patch(circle)

    # Add robot image
    robot_size = 0.40
    if robot_image_path_top:
        try:
            robot_img = Image.open(robot_image_path_top)

            # Rotate 90 degrees counter-clockwise
            robot_img = robot_img.rotate(90, expand=True)

            # Convert to RGBA for transparency
            if robot_img.mode != 'RGBA':
                robot_img = robot_img.convert('RGBA')

            extent = [-robot_size*vis.radius, robot_size*vis.radius,
                     -robot_size*vis.radius, robot_size*vis.radius]

            # Display with transparency to see heatmap through it
            ax2.imshow(robot_img, extent=extent, zorder=10, alpha=0.65)

            # Add subtle white border for depth
            # rect = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
            #                          robot_size*2*vis.radius, robot_size*2*vis.radius,
            #                          fill=False, edgecolor='white',
            #                          linewidth=2, zorder=11, alpha=0.7)
            # ax2.add_patch(rect)

            # Add shadow border
            # rect2 = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
            #                           robot_size*2*vis.radius, robot_size*2*vis.radius,
            #                           fill=False, edgecolor='black',
            #                           linewidth=1, zorder=11, alpha=0.3)
            # ax2.add_patch(rect2)

        except Exception as e:
            print(f"Could not load robot image: {e}")
            ax2.text(0, 0, '🤖', fontsize=50, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                    zorder=10)

    if show_cameras and vis.camera_positions is not None:
        ax2.scatter(vis.camera_positions[:, 0],
                   vis.camera_positions[:, 1],
                   c=vis.mse_values, cmap=cmap ,
                   s=15, edgecolors='black', linewidths=0.3,
                   zorder=5)

    ax2.set_aspect('equal')
    ax2.set_xlim([-vis.radius*1.1, vis.radius*1.1])
    ax2.set_ylim([-vis.radius*1.1, vis.radius*1.1])
    ax2.set_xlabel('X (m)', fontsize=9)
    ax2.set_ylabel('Y (m)', fontsize=9)
    ax2.tick_params(labelsize=8)
    # Title removed

    # Shared colorbar
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=mse_smooth.min(), vmax=mse_smooth.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('MSE', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined hemisphere with robot to {output_path}")


def plot_hemisphere_transparent_robot(vis, output_path, robot_image_path,
                                     cmap='hot_r', robot_alpha=0.5,
                                     robot_size=0.40, show_cameras=False, show_gradient=True):
    """
    Create a top-down view with highly transparent robot that blends into heatmap.
    Perfect for showing robot "sitting on top" of the hemisphere.

    Args:
        vis: HemisphereHeatmap object
        output_path: Path to save figure
        robot_image_path: Path to robot image
        cmap: Colormap name
        robot_alpha: Transparency level (0.0=invisible, 1.0=opaque)
        robot_size: Size of robot as fraction of hemisphere radius
                   Default 0.40 represents 19" robot (0.48m) in 1.2m hemisphere
        show_cameras: Whether to show camera positions
    """
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax = fig.add_subplot(111)

    # Create hemisphere mesh
    X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=60)

    # Interpolate accuracy
    mse_mesh = vis.interpolate_mse(X, Y, Z)

    from scipy.ndimage import gaussian_filter
    mse_smooth = gaussian_filter(mse_mesh, sigma=1.5)




        # Add gradient arrows in 2D
    # if show_gradient:
    #         ax.quiver(X[::step, ::step], Y[::step, ::step],
    #                   dx[::step, ::step], dy[::step, ::step],
    #                   alpha=0.5, width=0.003)



    # dx_raw, dy_raw = vis.compute_gradient(acc_smooth, X, Y, Z)
    # grad_mag = np.sqrt(dx_raw**2 + dy_raw**2)
    # max_mag = np.max(grad_mag)

    # if max_mag < 1e-12:
    #     ux = np.zeros_like(dx_raw)
    #     uy = np.zeros_like(dy_raw)
    # else:
    #     ux = dx_raw / (max_mag + 1e-12)
    #     uy = dy_raw / (max_mag + 1e-12)

    # # normalize magnitudes [0,1]
    # grad_norm = grad_mag / (max_mag + 1e-12)

    # # choose max arrow length in *hemisphere* units
    # L_max = 0.8 * vis.radius   # increase to 0.8*vis.radius if they still feel small

    # # components for quiver in hemisphere coords
    # dx = ux * grad_norm * L_max
    # dy = uy * grad_norm * L_max

    # # subsample grid so it’s not too busy
    # step = 8
    # Xq = X[::step, ::step]
    # Yq = Y[::step, ::step]
    # dxq = dx[::step, ::step] * 1.2   # scale arrows to match X_display/Y_display
    # dyq = dy[::step, ::step] * 1.2

    # ax.quiver(
    #     Xq, Yq,
    #     dxq, dyq,
    #     color="tab:blue",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1.0,
    #     width=0.004,
    #     headwidth=5,
    #     headlength=6,
    #     alpha=0.9,
    #     zorder=12,      # below robot (15) but above heatmap
    # )



    # Create heatmap with more levels for smoother look
    contourf = ax.contourf(X, Y, mse_smooth, levels=40, cmap=cmap)

    # Add camera positions behind robot
    if show_cameras and vis.camera_positions is not None:
        ax.scatter(vis.camera_positions[:, 0],
                  vis.camera_positions[:, 1],
                  c=vis.mse_values, cmap=cmap,
                  s=25, edgecolors='white', linewidths=0.5,
                  alpha=0.7, zorder=5)

    # Add robot image with transparency
    if robot_image_path:
        try:
            robot_img = Image.open(robot_image_path)

            # Rotate 90 degrees counter-clockwise
            robot_img = robot_img.rotate(90, expand=True)

            # Convert to RGBA
            if robot_img.mode != 'RGBA':
                robot_img = robot_img.convert('RGBA')

            # Make image more transparent
            img_array = np.array(robot_img)

            # Adjust alpha channel for entire image
            if img_array.shape[2] == 4:
                img_array[:, :, 3] = (img_array[:, :, 3] * robot_alpha).astype(np.uint8)

            robot_img_transparent = Image.fromarray(img_array)

            # *** SHIFT ROBOT UP IN Y DIRECTION ***
            y_offset = 0.2 * vis.radius  # Adjust this value

            # Calculate image extent with y-offset
            # extent = [left, right, bottom, top]
            extent = [
                -robot_size*vis.radius,              # left (x)
                robot_size*vis.radius,               # right (x)
                -robot_size*vis.radius + y_offset,   # bottom (y) - shifted up
                robot_size*vis.radius + y_offset     # top (y) - shifted up
            ]

            # Display transparent robot
            ax.imshow(robot_img_transparent, extent=extent, zorder=15)

            # Add very subtle glow effect around robot
            from matplotlib.patches import Circle
            glow = Circle((0, 0), robot_size * vis.radius * 1.1,
                         fill=True, facecolor='white',
                         alpha=0.15, zorder=14)
            ax.add_patch(glow)

            # Add thin white border
            rect = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
                                     robot_size*2*vis.radius, robot_size*2*vis.radius,
                                     fill=False, edgecolor='white',
                                     linewidth=2, zorder=16, alpha=0.5)
            # ax.add_patch(rect)

        except Exception as e:
            print(f"Could not load robot image: {e}")
            ax.text(0, 0, '🤖', fontsize=60, ha='center', va='center',
                   alpha=robot_alpha, zorder=15)
    # Add gradient arrows to show direction of improvement
    dx_raw, dy_raw = vis.compute_gradient(mse_smooth, X, Y, Z)
    grad_mag = np.sqrt(dx_raw**2 + dy_raw**2)
    max_mag = np.max(grad_mag)

    if max_mag > 1e-12:
        # Normalize gradient direction
        ux = dx_raw / (max_mag + 1e-12)
        uy = dy_raw / (max_mag + 1e-12)

        # Normalize magnitudes to [0, 1]
        grad_norm = grad_mag / max_mag

        # Maximum arrow length in hemisphere units
        L_max = 0.5 * vis.radius

        # Final arrow components
        dx = ux * grad_norm * L_max
        dy = uy * grad_norm * L_max

        # Subsample arrows so it's not too busy
        step = 8
        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            dx[::step, ::step],
            dy[::step, ::step],
            color="tab:blue",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.008,      # Increased from 0.004 - makes shaft thicker
            headwidth=7,      # Increased from 5 - makes head wider
            headlength=8,     # Increased from 6 - makes head longer
            alpha=0.9,
            zorder=5,  # Below robot (which is zorder=10)
        )
    # Add circle boundary
    circle = plt.Circle((0, 0), vis.radius, fill=False,
                       edgecolor='black', linewidth=2.5, zorder=20)
    ax.add_patch(circle)

    # Configure axes
    ax.set_aspect('equal')
    ax.set_xlim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_ylim([-vis.radius*1.1, vis.radius*1.1])
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    # Title removed - will be added externally as needed
    ax.tick_params(labelsize=10)
    # turnoff a;; ax
    ax.set_axis_off()

    # Add colorbar

    # vmin, vmax = contourf.get_clim()
    # ticks = np.linspace(vmin, vmax, 5)
    # cbar = plt.colorbar(contourf, ax=ax, ticks=ticks, shrink=0.8, pad=0.05,orientation='horizontal')
    # # cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, pad=0.05, orientation='horizontal')
    # # cbar.set_label('MSE', fontsize=11)
    # cbar.ax.tick_params(labelsize=35)



    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, pad=0.05, orientation='horizontal')
    cbar.locator = MaxNLocator(nbins=5)  # Aims for ~5 ticks with nice round values
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=35)


    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


    plt.close(fig)
    print(f"Saved transparent robot hemisphere to {output_path}")




if __name__ == "__main__":
    example_usage()

