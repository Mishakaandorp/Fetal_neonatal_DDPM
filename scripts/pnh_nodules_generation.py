import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, gaussian_filter
import random
import shutil
import os
# --subject $1 --number_nodules $2 --input_file $3 --output_file $4
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--number_nodules', type=int, default=50)
parser.add_argument('--outputdir', type=str, default='')
parser.add_argument('--input_file', type=str)
# parser.add_argument('--orig_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

sub = args.subject
number_nodules = args.number_nodules
outputdir = args.outputdir
input_file = args.input_file
# orig_file = args.orig_file
output_file = args.output_file

def load_nifti_image(image_path):
    """Load a 3D NIfTI image and return the image data."""
    img = nib.load(image_path)
    data = img.get_fdata()
    return img, data

def find_boundary_pixels(data, label, exclude_labels, min_distances, additional_exclude_masks):
    """Find the boundary pixels of the given label in the 3D image data, excluding those that are within min_distances pixels from any exclude_labels or additional_exclude_masks."""
    label_mask = data == label
    binary_mask = label_mask.astype(np.uint8)
    eroded_mask = binary_erosion(binary_mask)
    boundary_mask = binary_mask & ~eroded_mask
    boundary_coords = np.column_stack(np.where(boundary_mask))

    # Calculate the distance transforms for each exclude label and apply the minimum distance filter
    valid_coords = boundary_coords.copy()
    for exclude_label, min_distance in zip(exclude_labels, min_distances):
        exclude_mask = data == exclude_label
        distance_map = distance_transform_edt(~exclude_mask)
        valid_coords = [coord for coord in valid_coords if distance_map[tuple(coord)] >= min_distance]
        valid_coords = np.array(valid_coords)
    
    # Exclude coordinates that intersect with additional exclude masks
    for exclude_mask in additional_exclude_masks:
        valid_coords = [coord for coord in valid_coords if not exclude_mask[tuple(coord)]]
        valid_coords = np.array(valid_coords)
    
    return valid_coords

def get_random_boundary_coordinate(boundary_coords):
    """Choose a random coordinate from the boundary coordinates."""
    random_idx = random.randint(0, len(boundary_coords) - 1)
    return boundary_coords[random_idx]

def generate_sphere(center, radius, shape):
    """Generate a sphere of given radius around the center coordinate."""
    x, y, z = np.ogrid[
        -radius:radius+1, -radius:radius+1, -radius:radius+1
    ]
    sphere = (x**2 + y**2 + z**2) <= radius**2
    sphere_coords = np.column_stack(np.where(sphere))
    sphere_coords = sphere_coords - radius + center
    
    # If radius is 3 or larger, exclude the outermost pixels in all dimensions
    if radius >= 3:
        sphere_coords = [coord for coord in sphere_coords if all(
            abs(c - center[i]) < radius for i, c in enumerate(coord)
        )]
        sphere_coords = np.array(sphere_coords)
    
    # Filter out coordinates that are out of bounds
    valid_coords = (sphere_coords[:, 0] >= 0) & (sphere_coords[:, 0] < shape[0]) & \
                   (sphere_coords[:, 1] >= 0) & (sphere_coords[:, 1] < shape[1]) & \
                   (sphere_coords[:, 2] >= 0) & (sphere_coords[:, 2] < shape[2])
    return sphere_coords[valid_coords]

def smooth_sphere(data, center, radius):
    """Apply Gaussian smoothing to the sphere region."""
    sphere_data = np.zeros_like(data)
    sphere_coords = generate_sphere(center, radius, data.shape)
    for coord in sphere_coords:
        sphere_data[tuple(coord)] = 1

    # Apply Gaussian smoothing
    smoothed_sphere = gaussian_filter(sphere_data, sigma=radius / 2)
    return smoothed_sphere

def update_labelmap(data, sphere_data, new_label):
    """Update the label map data at the given coordinates to the new label."""
    mask = sphere_data > 0.5  # Threshold to create a binary mask from smoothed data
    data[mask] = new_label
    return data

def save_nifti_image(img, data, output_path):
    """Save the modified data to a new NIfTI image."""
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, output_path)

# Get bounding box midline
def save_boundary_coordinates_image(data_shape, boundary_coords, output_path, affine):
    """Save the boundary coordinates as a separate NIfTI image."""
    boundary_image = np.zeros(data_shape)
    for coord in boundary_coords:
        boundary_image[tuple(coord)] = 1  # Mark boundary coordinates with label 1
    boundary_img = nib.Nifti1Image(boundary_image, affine)
    nib.save(boundary_img, output_path)


def find_bounding_box_and_middle(data, label):
    """Find the bounding box and the middle point of that bounding box for a given label."""
    coords = np.column_stack(np.where(data == label))
    
    if coords.size == 0:
        raise ValueError(f"No coordinates found for label {label}")
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Bounding box coordinates
    bounding_box = (min_coords, max_coords)
    
    # Middle of the bounding box
    middle = (min_coords + max_coords) // 2
    
    return bounding_box, middle

def create_middle_point_image(data_shape, middle_point):
    """Create an image with the middle point marked."""
    middle_point_image = np.zeros(data_shape)
    middle_point_image[tuple(middle_point)] = 1  # Mark the middle point with 1
    return middle_point_image

def create_extended_mask(data, middle_point, label, x_extension=10, z_extension=20):
    """Create a mask that extends in the x and z directions by 'x_extension' and 'z_extension' pixels respectively and spans the y dimension of the label."""
    mask = np.zeros(data.shape)
    
    # Determine the y-dimension bounds for the label
    y_coords = np.where(data == label)[1]
    y_start, y_end = y_coords.min()-20, y_coords.max() + 1+20
    
    # Determine the x and z bounds with the specified extensions
    x_start, x_end = max(0, middle_point[0] - x_extension), min(data.shape[0], middle_point[0] + x_extension + 1)
    z_start, z_end = max(0, middle_point[2] - z_extension), min(data.shape[2], middle_point[2] + z_extension + 1)

    mask[x_start:x_end, y_start:y_end, z_start:z_end] = 1
    return mask

# Get 4th venticle mask
def combine_labels(label_map, label1, label2):
    """
    Combine two labels in the label map into a new binary map.
    """
    combined_map = (label_map == label1) | (label_map == label2)
    return combined_map

def dilate_combined_map(combined_map, iterations=5):
    """
    Dilate the combined map, only on the sides (x and y directions).
    """
    # Define a structuring element for dilation (sides only, not top/bottom)
    # Initialize a 3x3x3 array with all False values
    # Create an array of shape (3, 3, 3) filled with False
    # Initialize a 3x3x3 array with False

    # Define the 3D numpy array
    array = np.array([[[False, False, False],
                    [False, False, False],
                    [False, False, False]],

                    [[False, True, False],
                    [True, True, False],
                    [False, True, False]],

                    [[False, False, False],
                    [False, False, False],
                    [False, False, False]]])

    # print(array)
    
    # Perform dilation
    dilated_map = combined_map
    for _ in range(iterations):
        dilated_map = binary_dilation(dilated_map, structure=array)
    
    return dilated_map

def select_from_dilated_map(dilated_map, label_map, target_label):
    """
    Select regions from the label map where the dilated map is true and the label map has the target label.
    """
    selected_map = (label_map == target_label) & dilated_map
    return selected_map

#Generate for checks
image_path = f'{outputdir}/{input_file}'
output_path = f'{outputdir}/{output_file}'
output_path_copy = f'{outputdir}/{sub}/pnh_maps/{sub}_PNH_image_MAX_dilation.nii.gz'
boundary_output_path = f"{outputdir}/{sub}/pnh_maps/{sub}_dseg_boundary_pnh_{number_nodules}.nii.gz"
boundary_output_path_copy = f'{outputdir}/{sub}/pnh_maps/{sub}_boundary_image_PNH.nii.gz'
output_path_pixel = f"{outputdir}/{sub}/pnh_maps/{sub}_seg_output_pixel_bounding_boxpnh_{number_nodules}.nii.gz"
output_path_4th_ventricles = f"{outputdir}/{sub}/4th_ventricle/{sub}_seg_output_path_4th_ventricles.nii.gz"
path_midline_box = f'{outputdir}/{sub}/pnh_maps/{sub}_seg_orig_derotated_bounding_box_midline.nii.gz'
path_midline_box_copy = f"{outputdir}/{sub}/pnh_maps/{sub}_seg_midline.nii.gz"


os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(output_path_copy), exist_ok=True)
os.makedirs(os.path.dirname(boundary_output_path), exist_ok=True)
os.makedirs(os.path.dirname(boundary_output_path_copy), exist_ok=True)
os.makedirs(os.path.dirname(output_path_pixel), exist_ok=True)
os.makedirs(os.path.dirname(output_path_4th_ventricles), exist_ok=True)
os.makedirs(os.path.dirname(path_midline_box_copy), exist_ok=True)

# Load the image
img, data = load_nifti_image(image_path)
label = 4
new_label = 2
exclude_labels = [0, 1, 2, 6]
min_distances = [4, 4, 4, 4]

label_DGM = 6

if sub in ['xx']:
    # These subjects do have an angled skewed brain in the axial slice. 
    # Hence, these are processed by a seperate script.
    _, midline_box_mask = load_nifti_image(path_midline_box)
    print('rotated_map_first_done_loaded')
else:
    bounding_box, middle = find_bounding_box_and_middle(data, label_DGM)
    print(f"Bounding box for label {label_DGM}: {bounding_box}")
    print(f"Middle of the bounding box: {middle}")

    # Create an image with the middle point marked
    middle_point_image = create_middle_point_image(data.shape, middle)

    # Create a mask with the specified extension around the middle point
    midline_box_mask = create_extended_mask(data, middle, label_DGM, x_extension=15, z_extension=30)

    # Save the middle point image
    save_nifti_image(img, middle_point_image, output_path_pixel)

    # Save the extended mask
    save_nifti_image(img, midline_box_mask, path_midline_box)

## Get 4th ventricle mask
combined_mask_cerebellum_brainstem = combine_labels(data, label1=5, label2=7)
combined_mask_cerebellum_brainstem_dilated = dilate_combined_map(combined_mask_cerebellum_brainstem, iterations=5)
combined_mask_cerebellum_brainstem_dilated_selected_map_label_4 = select_from_dilated_map(combined_mask_cerebellum_brainstem_dilated, data, 4)

# Save the modified label map for 4th ventricles
save_nifti_image(img, combined_mask_cerebellum_brainstem_dilated_selected_map_label_4, output_path_4th_ventricles)

# Exclude regions from boundary calculation
additional_exclude_masks = [combined_mask_cerebellum_brainstem_dilated_selected_map_label_4, midline_box_mask]
# additional_exclude_masks =[]
boundary_coords = find_boundary_pixels(data, label, exclude_labels, min_distances, additional_exclude_masks)

if len(boundary_coords) == 0:
    print("No boundary coordinates found for the given label.")
else:
    # Save the boundary coordinates image
    save_boundary_coordinates_image(data.shape, boundary_coords, boundary_output_path, img.affine)
    
    # Loop over number nodules
    for i in range(number_nodules):
        print(f'Generate nodule {i} / {number_nodules} in the map')
        random_boundary_coord = get_random_boundary_coordinate(boundary_coords)
        # print(f"Random boundary coordinate: {random_boundary_coord}")

        Var_radius = random.randint(4, 5)  # Radius can be 4 or 5
        smoothed_sphere = smooth_sphere(data, random_boundary_coord, Var_radius)
        
        # print(f"Smoothed sphere around {random_boundary_coord} with radius {Var_radius} created.")
        
        # Update the label map
        data = update_labelmap(data, smoothed_sphere, new_label)

    # Save the modified label map
    save_nifti_image(img, data, output_path)

# Coppying for easy visualization
shutil.copyfile(path_midline_box, path_midline_box_copy)
shutil.copyfile(output_path, output_path_copy)
shutil.copyfile(boundary_output_path, boundary_output_path_copy)
