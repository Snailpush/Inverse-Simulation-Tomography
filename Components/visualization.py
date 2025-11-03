import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os
from skimage.restoration import unwrap_phase
import cv2
import tqdm


import pyvista as pv
from matplotlib import colormaps


from Components import utils



################################################################
# Note for Visualization:
#
# The 3D object coordinate system is defined as:
#
#     y
#     ^      z
#     |    /
#     |  /
#     |/
#     +-----------> x
# with voxel_object[x, y, z]
#
# When visualizing a slice with:
#     slice = voxel_object[:, :, z]
#     plt.imshow(slice)
#
# Note:
# - This slice has shape (x, y)
# - Matplotlib's imshow interprets 2D arrays as:
#
#     +-----------> x (columns = axis 1 = y in object)
#     |
#     |
#     |
#     v
#     y (rows = axis 0 = x in object)
#
# - The image origin (0,0) is by default at the **top-left**
#   (i.e., increasing row index moves downward).
#
# Therefore:
# - The object’s **x-axis** is mapped to **image vertical (rows)**.
# - The object’s **y-axis** is mapped to **image horizontal (columns)**.
# - The object’s **z-axis** increases into the screen (away from viewer).   
#
################################################################



def get_label_units(units):
    """Helper function to return appropriate axis labels based on the units."""
    if units == 1e-6:
        return 'X (um)', 'Y (um)', 'Z (um)' 
    elif units == 1e-3:
        return 'X (mm)', 'Y (mm)', 'Z (um)' 
    else:
        return 'X (m)', 'Y (m)', 'Z (um)' 
    

def get_slice_image(RI_distribution, axis="z", idx=250):
    """Get a specific slice along an axis at index position"""
    axis_map = {
        "z": RI_distribution[:, :, idx],
        "x": RI_distribution[idx, :, :],
        "y": RI_distribution[:, idx, :],
    }
    
    try:
        return axis_map[axis]
    except KeyError:
        raise ValueError("Wrong axis. Must be one of {'x', 'y', 'z'}.")
    

def get_extent(support, axis="z", origin="top-left"):
    axis_indices = {
        "x": (1, 2), # y, z
        "y": (0, 2), # x, z
        "z": (0, 1)  # x, y
    }

    if axis not in axis_indices:
        raise ValueError("Wrong Axis. Limited to 'x', 'y', 'z' axis.")
    if origin not in ["top-left", "bottom-left"]:
        raise ValueError("Wrong origin. Limited to 'top-left' or 'bottom-left'.")

    i, j = axis_indices[axis]

    if origin == "top-left":
        extent = [0, support[i], support[j], 0]
    else:  # bottom-left
        extent = [0, support[i], 0, support[j]]

    return extent


def get_titles(title, gt_title, pose, gt_pose, pose_unit="um"): 
    """Helper function to format titles with pose information
    Used in comparison plots"""

    position = utils.round_list((pose["Position"]).detach().cpu().tolist(), 3)
    axis = utils.round_list(pose["Axis"].detach().cpu().tolist(), 3)
    angle = round(pose["Angle"].detach().cpu().item(), 3)
    
    title += f"Position: {position} {pose_unit} \n"
    title += f"Axis: {axis} \n"
    title += f"Angle: {angle}° \n"

    if gt_pose is not None:
    
        gt_position = utils.round_list((gt_pose[0]*1e6).detach().cpu().tolist(), 3)
        gt_axis = utils.round_list(gt_pose[1].detach().cpu().tolist(), 3)
        gt_angle = round(gt_pose[2].detach().cpu().item(), 3)
        
        gt_title += f"Position: {gt_position} um \n"
        gt_title += f"Axis: {gt_axis} \n"
        gt_title += f"Angle: {gt_angle}° \n"


    return title, gt_title 
    
################################################################





################################################################


def base_plot(image, support, title, axis_labels, extent=None, cmap="jet", vmin=None, vmax=None, grid=False, origin="top-left"):
    """General Plotting for single 2D Images"""
    
    fig, ax = plt.subplots(figsize=(6.4,4.8), dpi=150)

    # Default plt imshow
    if origin == "top-left":
         
        if extent is None:
            extent = [0, support[0],  support[1], 0]

        # Origin Top-Left x-axis runs downwards
        im = ax.imshow(image, cmap=cmap, extent=extent,
                   vmin=vmin, vmax=vmax)
        
    
    # Easier Human Readable
    elif origin == "bottom-left":

        if extent is None:
            extent = [0, support[0], 0, support[1]]

    
        # Origin bottom-left, x-axis runs across
        im = ax.imshow(image.T, cmap=cmap, 
                       extent=extent, origin="lower",
                       vmin=vmin, vmax=vmax)
        

    
    # Colorbar
    cbar = fig.colorbar(im)
    cbar.formatter = ticker.FuncFormatter(lambda x, _: f"{x:.7}")
    cbar.update_ticks()

   
    # Plot Title
    ax.set_title(title)

    # Axis Lables - (see note)
    ax.set_xlabel(axis_labels["x-axis"])
    ax.set_ylabel(axis_labels["y-axis"])
    
    # Grid Lines - every 5 units
    if grid:
        plt.xticks(range(0, int(support[0]) + 1, 5))
        plt.yticks(range(0, int(support[1]) + 1, 5))
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    return fig, ax 






def comparison_plot(images, support, titles, axis_labels, extent=None, cmap=None, vmin=None, vmax=None, grid=False, origin="top-left"):
    """Base plot to display multiple images side-by-side"""
    fig, axs = plt.subplots(1, len(images), figsize=(16, 8))
    

    for i, (image, title) in enumerate(zip(images, titles)):
        
        # Default plt imshow
        if origin == "top-left":
            
            if extent is None:
                extent = [0, support[0] , support[1], 0]

            im = axs[i].imshow(image, cmap=cmap, extent=extent,
                   vmin=vmin, vmax=vmax)

        
         # Easier Human Readable
        elif origin == "bottom-left":

            if extent is None:
                extent = [0, support[0], 0, support[1]]

            # Origin bottom-left, x-axis runs across
            im = axs[i].imshow(image.T, cmap=cmap, 
                           extent=extent, origin="lower",
                           vmin=vmin, vmax=vmax)
            
        # Axis Labels - (see note)
        axs[i].set_xlabel(axis_labels["x-axis"])
        axs[i].set_ylabel(axis_labels["y-axis"])

        
        # Colorbar
        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        
        # Plot Title
        axs[i].set_title(title)
        
        # Grid Lines
        if grid:
            axs[i].set_xticks(range(0, int(support[0]) + 1, 5))
            axs[i].set_yticks(range(0, int(support[1]) + 1, 5))
            axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
    return fig, axs


 

def sim_space_render(data, opacity, title, output_file, colormap='jet'):
    """Screenshots a image of a 3D Volume Render
    Displayed Axis are a right handed coordinate system while the simulation space is a left handed coordinate system."""
    
    
    grid = pv.wrap(data)

    cmap = colormaps[colormap]

    # Create a volume plot
    plotter = pv.Plotter(off_screen=True)
    plotter.add_title(title)
    plotter.add_volume(grid, cmap=cmap, opacity=opacity, n_colors=256)
    plotter.show_axes() 

    plotter.render()  # Important: must explicitly call render before screenshot
    plotter.screenshot(output_file)
    plotter.close()




def rgb_plot(x, rgb_array, title="", labels=("", "")):
    """Plot RGB values as scatter plot
    Three scatter plots in one figure for x, y, z values of rgb_array colored in r, g, b respectively"""
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(x, rgb_array[:,0], marker=".", color="r", label="$x$")
    ax.scatter(x, rgb_array[:,1], marker=".", color="g", label="$y$")
    ax.scatter(x, rgb_array[:,2], marker=".", color="b", label="$z$")

    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xticks(np.arange(x[0], x[-1], 50).astype(int))

    return fig, ax

def expected_scatter_plot(x,y, title="", labels=("",""), expected_min=0, expected_max=360, ignore_first=False):
    """Scatter plot with expected line"""

    fig, ax = scatter_plot(x,y, title, labels, ignore_first)

    ax.plot(x, np.linspace(expected_min, expected_max, len(x)), color="r", linestyle="--", alpha =0.5, label="Expected")

    return fig, ax

def scatter_plot(x, y, title="", labels=("",""), ignore_first=False):
    """Simple scatter plot"""
    
    if ignore_first:
        x = x[1:]
        y = y[1:]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(x, y, marker=".")

    ax.set_title(title)
    plt.grid(True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    if len(x) > 0:
        ax.set_xticks(np.arange(x[0], x[-1], 50).astype(int))

    return fig, ax



def plot_loss(losses, title, log_scale=False):
    """Plot loss curve -- Simple line plot"""

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(range(len(losses)), losses)
    ax.grid(True)
    
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

    
    return fig














def write_video(image_folder, dest_dir, file_name):
    """Concatenates all image files in a folder to a video"""
    files = [f for f in os.listdir(image_folder)]

    # Read first image to get size
    frame = cv2.imread(os.path.join(image_folder, files[0]))
    height, width, _ = frame.shape

    # Define video writer
    out = cv2.VideoWriter(os.path.join(dest_dir, file_name), cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 0)

    # Write frames
    with tqdm.tqdm(total=len(files), desc="Frames", unit="frame") as pbar:
        for fname in files:
            img = cv2.imread(os.path.join(image_folder, fname))
            img = cv2.putText(img,fname, (2100, 50), font, font_scale, color, thickness)
            out.write(img)

            pbar.update(1)

    out.release()



def write_video_from_folders(dirs, prefix, out_dir, video_name, fps=30, name="dir"):
    """
    Creates a video by taking the first image from each folder that starts with a given prefix.
    
    Args:
        dirs (list[str]): List of folder paths to search for images.
        prefix (str): Prefix to match at the start of the image filenames.
        out_dir (str): Output directory to save the final video.
        video_name (str): Name of the resulting video file (e.g., 'output.avi').
        fps (int, optional): Frames per second for the output video. Default is 30.
    """
    # Validate input folders
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise ValueError("No valid directories provided.")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Sort directories for consistent video order
    dirs.sort()

    # Find the first matching file in the first directory
    first_dir = dirs[0]
    first_file_candidates = [f for f in os.listdir(first_dir) if f.startswith(prefix)]
    if not first_file_candidates:
        raise FileNotFoundError(f"No file starting with '{prefix}' found in {first_dir}")

    first_file = os.path.join(first_dir, first_file_candidates[0])

    # Read the first image to determine video dimensions
    first_img = cv2.imread(first_file)
    if first_img is None:
        raise ValueError(f"Could not read image: {first_file}")
    height, width, _ = first_img.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(out_dir, video_name)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Font settings for overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    color = (0, 0, 0)  # Black text

    skipped_dirs = []

    # Iterate through all directories and add frames
    with tqdm.tqdm(total=len(dirs), desc="Folders", unit="folder") as pbar:
        for directory in dirs:
            # Find the first file that starts with the prefix
            matches = [f for f in os.listdir(directory) if f.startswith(prefix)]
            if not matches:
                skipped_dirs.append(directory)
                pbar.update(1)
                continue

            file_name = matches[0]
            img_path = os.path.join(directory, file_name)
            img = cv2.imread(img_path)

            if img is None:
                skipped_dirs.append(directory)
                pbar.update(1)
                continue

            if name == "dir":
                text = os.path.basename(directory)
            else:
                text = file_name

            # Add overlay text with the file name
            img = cv2.putText(img, text, (50, height - 25), font, font_scale, color, thickness)

            # Write the frame
            out.write(img)
            pbar.update(1)

    out.release()

    if skipped_dirs:
        print("Skipped directories due to missing or unreadable images:")
        for d in skipped_dirs:
            print(f" - {d}")




def save_plot(fig, path, file_name):
    """Save figure to location"""
    os.makedirs(path, exist_ok=True)
    
    fig.savefig(os.path.join(path, file_name), dpi=150)
    plt.close(fig)