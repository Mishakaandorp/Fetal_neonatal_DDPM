import nibabel as nib
import numpy as np
import os
import argparse
from scipy.ndimage import binary_dilation, gaussian_filter

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--shrink_factor', type=float)
parser.add_argument('--outputdir', type=str)
parser.add_argument('--input_file', type=str)
parser.add_argument('--orig_file', type=str)
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

# Arguments
sub = args.subject
shrink_factor = args.shrink_factor
outputdir = args.outputdir
input_file = args.input_file
orig_file = args.orig_file
output_file = args.output_file

# Find the left/right top and bottom most coordinate of a label (7) to use for transformation
def find_leftmost_coordinates(label_map):
    coordinates = {}

    # Find the bottom slice containing label 7
    for z in range(label_map.shape[2] - 1, -1, -1):
        slice_data = label_map[:, :, z]
        if np.any(slice_data == 7):
            bottom_slice_z = z
            break
    
    # Find the leftmost coordinate in the bottom slice containing label 7
    bottom_slice_data = label_map[:, :, bottom_slice_z]
    label_7_coordinates = np.argwhere(bottom_slice_data == 7)
    leftmost_coordinate_bottom = label_7_coordinates[np.argmax(label_7_coordinates[:, 1])]
    coordinates['bottom'] = (bottom_slice_z, leftmost_coordinate_bottom)

    # Find the top slice containing label 7
    for z in range(label_map.shape[2]):
        slice_data = label_map[:, :, z]
        if np.any(slice_data == 7):
            top_slice_z = z
            break
    # Find the leftmost coordinate in the top slice containing label 7
    top_slice_data = label_map[:, :, top_slice_z]
    label_7_coordinates = np.argwhere(top_slice_data == 7)
    leftmost_coordinate_top = label_7_coordinates[np.argmax(label_7_coordinates[:, 1])]
    coordinates['top'] = (top_slice_z, leftmost_coordinate_top)
    
    return coordinates

#  process files
def process_file(input_file):
    # breakpoint()
    label_map_nii = nib.load(input_file)
    label_map = label_map_nii.get_fdata()
    coordinates = find_leftmost_coordinates(label_map)

    return label_map, coordinates

# Shift file transcript
def shift_label_map(label_map, shift_amount):
    shifted_label_map = np.zeros_like(label_map)
    for z in range(label_map.shape[2]):
        # Shift the pixels to the left by the shift_amount for each slice
        shift_amount_increasing = int(shift_amount * z)
        for x in range(label_map.shape[0]):
            # breakpoint()
            for y in range(label_map.shape[1] - shift_amount_increasing):
                shifted_label_map[x, y, z] = label_map[x , y - shift_amount_increasing, z]
    return shifted_label_map

# breakpoint()
# Define input files
input_file_trans = f"{input_file}"
input_file_orig = f"{outputdir}/{orig_file}"


# Process the files and print the coordinates 
label_map_trans, coordinates_trans = process_file(input_file_trans)

label_map_orig, coordinates_orig = process_file(input_file_orig)

# Calculate differences
diff_label_trans_top_bottom = abs(coordinates_trans['bottom'][0] - coordinates_trans['top'][0])
diff_label_trans_top_bottom_left = abs(coordinates_trans['bottom'][1][1] - coordinates_trans['top'][1][1])

diff_label_orig_top_bottom = abs(coordinates_orig['bottom'][0] - coordinates_orig['top'][0])
diff_label_orig_top_bottom_left = abs(coordinates_orig['bottom'][1][1] - coordinates_orig['top'][1][1])

# print("Difference in slices (transformed):", diff_label_trans_top_bottom)
# print("Difference in leftmost coordinates (transformed):", diff_label_trans_top_bottom_left)
# print("Difference in slices (original):", diff_label_orig_top_bottom)
# print("Difference in leftmost coordinates (original):", diff_label_orig_top_bottom_left)

# Find total change
change_orig_trans = abs(diff_label_orig_top_bottom_left - diff_label_trans_top_bottom_left)
# Find how much it should change per slice in the y direction
change_per_slice = change_orig_trans / diff_label_trans_top_bottom

# print("Change per slice in leftmost coordinate:", change_per_slice)


# Shift the transformed label map wit hchanges per slice
print(f'Shifting the shinked transformed label ..')
shifted_label_map_trans = shift_label_map(label_map_trans, change_per_slice)

#  Initialize again..
input_file_orig_nii = nib.load(input_file_orig)
file_orig = input_file_orig_nii.get_fdata()

# Save the shifted label map for visualization..
output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_rec-mial_dseg_trans_shifted_Y_scaling_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(shifted_label_map_trans, input_file_orig_nii.affine), output_file_trans_shifted)

# Save the original file..
output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_rec-mial_dseg_trans_ORIG_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(file_orig, input_file_orig_nii.affine), output_file_trans_shifted)

# Functions for transformation shifted labelmap
#  Do alternative replacement of labels
def replace_label(original_label_map, another_label_map):
    # Copy the original label map to avoid modifying the original array
    replaced_label_map = np.copy(original_label_map)
    
    # Find the indices where label 7 occurs in the original label map
    indices_7_another_label_map = (another_label_map == 7)
    indices_5_another_label_map = (another_label_map == 5)
    indices_10_another_label_map = (another_label_map == 10)
    indices_5_original_label_map = (original_label_map == 5)
    indices_7_original_label_map = (original_label_map == 7)
    
    # Replace label 7 in the original label map with label 7 from another label map
    original_label_map[indices_7_original_label_map] = 1
    original_label_map[indices_5_original_label_map] = 1
    original_label_map[indices_7_another_label_map] = 7
    original_label_map[indices_5_another_label_map] = 5
    original_label_map[indices_10_another_label_map] = 10
    
    # breakpoint()
    return original_label_map


def combine_labels(label_map, label1, label2):
    """
    Combine two labels in the label map into a new binary map.
    """
    combined_map = (label_map == label1) | (label_map == label2)
    return combined_map

# Define 4th ventricles..
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

# Shifted map define 4th ventricles
shifted_label_map_trans_combined_5_7 = combine_labels(shifted_label_map_trans, label1=5, label2=7)
shifted_label_map_trans_combined_5_7_dilated = dilate_combined_map(shifted_label_map_trans_combined_5_7, iterations=5)
shifted_label_map_trans_selected_map_label_4 = select_from_dilated_map(shifted_label_map_trans_combined_5_7_dilated, shifted_label_map_trans, 4)
shifted_label_map_trans[shifted_label_map_trans_selected_map_label_4]=10

# Original map define 4th ventricles, set similar to eCSF
copy_file_org = np.copy(file_orig)
file_orig_combined_5_7 = combine_labels(copy_file_org, label1=5, label2=7)
file_orig_combined_5_7_dilated = dilate_combined_map(file_orig_combined_5_7, iterations=5)
file_orig_selected_map_label_4 = select_from_dilated_map(file_orig_combined_5_7_dilated, copy_file_org, 4)
copy_file_org[file_orig_selected_map_label_4]=1

# replace labelmaps
replaced_label_map = replace_label(copy_file_org, shifted_label_map_trans)

#  Initialize again..
label_map_trans_nii = nib.load(input_file_trans)
label_map_trans = label_map_trans_nii.get_fdata()

# Save the shifted label map for visualization..
output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_shifted_label_map_trans_selected_map_label_4_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(shifted_label_map_trans, label_map_trans_nii.affine), output_file_trans_shifted)

output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_file_orig_selected_sublabel_4_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(copy_file_org, label_map_trans_nii.affine), output_file_trans_shifted)

# breakpoint()
output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_rec-replaced_label_map_scaling_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(replaced_label_map, label_map_trans_nii.affine), output_file_trans_shifted)

# Find the middle coordinate of the bottom and top slice to replace the shrinked cerebellum with.
def find_middle_coordinates(label_map, label):
    coordinates = {}

    # Find the bottom slice containing label label
    for z in range(label_map.shape[2] - 1, -1, -1):
        slice_data = label_map[:, :, z]
        if np.any(slice_data == label):
            bottom_slice_z = z
            break

    # Find the middle coordinate in the bottom slice containing label (label)
    bottom_slice_data = label_map[:, :, bottom_slice_z]
    label_label_coordinates = np.argwhere(bottom_slice_data == label)
    middle_coordinate_bottom = label_label_coordinates[np.argmax(label_label_coordinates[:, 1])]
    coordinates['bottom'] = (bottom_slice_z, tuple(middle_coordinate_bottom.astype(int)))

    # Find the top slice containing label label
    for z in range(label_map.shape[2]):
        slice_data = label_map[:, :, z]
        if np.any(slice_data == label):
            top_slice_z = z
            break

    # Find the middle coordinate in the top slice containing label label
    top_slice_data = label_map[:, :, top_slice_z]
    label_label_coordinates = np.argwhere(top_slice_data == label)
    middle_coordinate_top = label_label_coordinates[np.argmax(label_label_coordinates[:, 1])]
    coordinates['top'] = (top_slice_z, tuple(middle_coordinate_top.astype(int)))

    return coordinates

def shift_label(label_map, zeros_map, label, shift_x, shift_y, shift_z):
    # Find the coordinates of the label
    label_coordinates = np.argwhere(label_map == label)
    
    # Shift the label along the x-axis and y-axis
    shifted_coordinates = label_coordinates + [shift_x, shift_y, shift_z]
    
    # Create a new label map with the shifted label
    for coord in shifted_coordinates:
        y, x,z = coord

        # Check if the shifted coordinate is within the bounds of the label map
        if 0 <= y < label_map.shape[0] and 0 <= x < label_map.shape[1] and 0 <= z < label_map.shape[2]:
            zeros_map[y, x, z] = label
    
    return zeros_map

# print(f'Find middle cooridinates orignal and replaced label map ..')
coordinates_shifted_map = find_middle_coordinates(replaced_label_map,7)

coordinates_orig = find_middle_coordinates(file_orig,7)
# print(f"Coordinates for original file:")
# print(f"Bottom slice: {coordinates_orig['bottom'][0]}, middle coordinate: {coordinates_orig['bottom'][1]}")
# print(f"Top slice: {coordinates_shifted['top'][0]}, Leftmost coordinate: {coordinates_shifted['top'][1]}")

input_file_orig_nii = nib.load(input_file_orig)
file_orig = input_file_orig_nii.get_fdata()

# Define difference x and Y cooridnate
difference_x_orig = coordinates_orig['bottom'][1][0] - coordinates_shifted_map['bottom'][1][0]
difference_y_orig = coordinates_orig['bottom'][1][1] - coordinates_shifted_map['bottom'][1][1]

# create zeros map 
print(f'Shift the replaced labelmap with a transformation based on the original labelmap .. ')

zeros_map = np.zeros_like(replaced_label_map)
# Paste the shifted labels
shifted_label_map = shift_label(replaced_label_map, zeros_map, 5, difference_x_orig, difference_y_orig+2,1)
shifted_label_map = shift_label(replaced_label_map, shifted_label_map, 7, difference_x_orig, difference_y_orig+2,1)
shifted_label_map = shift_label(replaced_label_map, shifted_label_map, 10, difference_x_orig, difference_y_orig+2,1)

# Save the shifted label map for visualization..
output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_rec-shifted_label_sub{shrink_factor}_almost_final.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(shifted_label_map, label_map_trans_nii.affine), output_file_trans_shifted)
# shifted_label_map = shift_label(shifted_label_map, 5, difference_x_orig, difference_y)

# Merge the labelmaps of the original map and the transformed map
def merge_label_maps(label_map1, label_map2):
    # Set label 8 in label_map1 to 1
    label_map1[label_map1 == 8] = 1
    label_map1[label_map1 == 7] = 1
    label_map1[label_map1 == 5] = 1
    label_map1[label_map1 == 9] = 1
    
    # Add label map 7 from label_map2
    label_map1[label_map2 == 7] = 7
    # Add label map 7 from label_map2
    label_map1[label_map2 == 5] = 5
    label_map1[label_map2 == 10] = 4
    # label_map1[label_map2 == 4] = 4
    
    return label_map1

merge_label_map= merge_label_maps(label_map_orig, shifted_label_map)

print(f'Merge and replace the original labelmap the replaced labelmap with a transformation based on the original labelmap .. ')

# !!!!!!!!!!!!!!!!!!!!!
# Presently the cerebellum (label 5) is also shrinked, whcih is not done in a realisitic way (only x direction) 
# Now, the cerebellum will be dilated with the following functions.. 
# With the original cerebellum shrinkage script (seperate script), the cerebellum will be shrinked again.. 
# !!!!!!!!!!!!!!!!!!!!!!!!!!

from scipy.ndimage import binary_dilation, gaussian_filter

def find_left_and_right_most_coordinates(label_map):
    coordinates = {}

    # Find all coordinates with label 7
    label_7_coordinates = np.argwhere(label_map == 5)

    # Find the leftmost coordinate (min x value)
    leftmost_coordinate_y = label_7_coordinates[np.argmin(label_7_coordinates[:, 1])]
    
    # Find the rightmost coordinate (max x value)
    rightmost_coordinate_y = label_7_coordinates[np.argmax(label_7_coordinates[:, 1])]
    
    coordinates['leftmost_y'] = leftmost_coordinate_y
    coordinates['rightmost_y'] = rightmost_coordinate_y


    return coordinates

def dilate_label_2d_only(labelmap, label, max_iterations=3):
    
    dilated_labelmap = np.copy(labelmap)
    dilated_labelmap_copy = np.copy(labelmap)
    top_slice, bottom_slice, middle_slice = None, None, None
    
    # Define structuring elements
    half_cross_right = np.array([[0, 0, 0],
                                 [1, 1, 0],
                                 [0, 0, 0]], dtype=bool)
    
    full_cross = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)
    
    z_indices = np.any(labelmap == label, axis=(0, 1)).nonzero()[0]
    if z_indices.size > 0:
        top_slice = z_indices[0]
        bottom_slice = z_indices[-1]
        middle_slice = z_indices[len(z_indices) // 2]
        
        # print(f"Top slice index for label {label}: {top_slice}")
        # print(f"Middle slice index for label {label}: {middle_slice}")
        # print(f"Bottom slice index for label {label}: {bottom_slice}")

        for z in range(labelmap.shape[2]):
            # Isolate the label in the 2D slice
            slice_2d = (labelmap[:, :, z] == label)

            iterations = max_iterations * np.exp(-(z - middle_slice) ** 2 / (2 * (max_iterations / 2) ** 2))
            iterations = int(max(iterations, 1))  # Ensure minimum of 1 iteration
            # print(f'iterations {iterations}')
            # Apply binary dilation to the 2D slice iteratively and update the labelmap at each step
            dilated_slice = slice_2d
            for i in range(iterations):
                if (i + 1) % 4 == 0:
                    struct_element = full_cross
                else:
                    struct_element = half_cross_right
                dilated_slice = binary_dilation(dilated_slice, structure=struct_element)
                # Exclude labels 5 and 7 from overwriting the new dilation
                dilated_labelmap[:, :, z][dilated_slice & ~(labelmap[:, :, z] == 5) & ~(labelmap[:, :, z] == 7)] = label


        
    # Apply Gaussian smoothing to the dilated regions with label 5 (cerebellum)
    smooth_labelmap = np.zeros(dilated_labelmap.shape)
    # dilated_labelmap_copy = np.copy(dilated_labelmap)
    smooth_labelmap[dilated_labelmap == 5] = 1  # Create a binary mask for the label

    sigma = 6 # defined as good smoothing
    smooth_labelmap = gaussian_filter(smooth_labelmap, sigma=sigma)  # Apply Gaussian filter with sigma
    
    smooth_labelmap = (smooth_labelmap > 0.3).astype(dilated_labelmap.dtype)  # Threshold to create binary mask
    
    dilated_labelmap[dilated_labelmap == 5] = 1  # Create a binary mask for the label
    dilated_labelmap[smooth_labelmap == 1] = 5  # Create a binary mask for the label
    
    # Copy everything from dilated_labelmap_copy except for labels 5 and 7
    mask = (dilated_labelmap_copy != 5) & (dilated_labelmap_copy != 1)
    dilated_labelmap[mask] = dilated_labelmap_copy[mask]

    return dilated_labelmap

# print(f'Dilate cerebellum based on the shrinked shape and the original shape .. ')

# Initialize maps..
input_file_orig_nii = nib.load(input_file_orig)
file_orig = input_file_orig_nii.get_fdata()

# Find coordinates
coordinates_shifted_label = find_left_and_right_most_coordinates(merge_label_map)
coordinates_orig_label = find_left_and_right_most_coordinates(file_orig)
diff_x_sifted = coordinates_shifted_label['rightmost_y'] - coordinates_shifted_label['leftmost_y']
diff_x_orig = coordinates_orig_label['rightmost_y'] - coordinates_orig_label['leftmost_y']

difference_shift_orig = diff_x_orig[1] - diff_x_sifted[1]
# 15/21 seems to be an appropriate ratio
max_iterations =int(difference_shift_orig * (15/21))           

label = 5
# max_iterations = 15
structure = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=bool)

# If the max iterations >0 then dilate else, not neccessary
if max_iterations >0:  
    dilate_cerebellum_label_map = dilate_label_2d_only(merge_label_map, label, max_iterations)
else:
    dilate_cerebellum_label_map=merge_label_map# breakpoint()

output_file_trans_shifted = f"{outputdir}/{sub}/maps_scaling/{sub}_rec-merge_label_map_scaling_2_{shrink_factor}.nii.gz"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(merge_label_map, label_map_trans_nii.affine), output_file_trans_shifted)

# Create output!
output_file_trans_shifted = f"{output_file}"
os.makedirs(os.path.dirname(output_file_trans_shifted), exist_ok=True)
nib.save(nib.Nifti1Image(dilate_cerebellum_label_map, label_map_trans_nii.affine), output_file_trans_shifted)
