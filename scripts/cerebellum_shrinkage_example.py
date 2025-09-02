import os
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import affine_transform

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--shrink_factor_cerebellum', type=float, default=None)
parser.add_argument('--scaling_factor_brainstem', type=float, default=None)
parser.add_argument('--outputdir', type=str, default='')
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()
map_ = args.subject
shrink_factor = args.shrink_factor_cerebellum
scaling_factor_brainstem = args.scaling_factor_brainstem
outputdir = args.outputdir
input_file = args.input_file
output_file = args.output_file

print(input_file)

# Alternative calculation for scaling cerebellum. 
if shrink_factor is None and scaling_factor_brainstem is not None and scaling_factor_brainstem > 0:
    # Scaling cerebellum is between 1-2.
    # Scaling brainstem is between 0.5-1.
    # Therefore, if the scaling brainstem is occuring, 
    # do the following scaling to also scale the cerebellum

    shrink_factor = -2 * scaling_factor_brainstem + 3
    print(f'New shrink factor {shrink_factor}, calculated as -2 * scaling_factor_brainstem + 3, where scaling_factor_brainstem = {scaling_factor_brainstem}')


import SimpleITK as sitk

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform

def shrink_label_map_nii(label_map_nii, shrink_factor=0.5):
    # Load the label map data
    label_map = label_map_nii.get_fdata()
    
    # Find the coordinates of label 5 = cerebellum
    label_5_indices = np.argwhere(label_map == 5)
    if len(label_5_indices) > 0:
        center = np.mean(label_5_indices, axis=0)
    else:
        print('NOO LABEL 5 = Cerebellum FOUND')
        breakpoint()
    center_round = np.round(center)
    slice_data = label_map[:, :, int(center_round[2])]
    # Find the point where the cerebllum reaches label 7 (= brainstem)
    coordinates = np.argwhere(slice_data == 7)
    if coordinates.size == 0:
        # If label 7 is not found at the center of the cerebelum, 
        # find the bottom (or top in this case) slice containing label 5
        for z in range(label_map.shape[2] - 1, -1, -1):
            slice_data = label_map[:, :, z]
            if np.any(slice_data == 5):
                bottom_slice_z = z
                break
        
        # Find the leftmost coordinate in the bottom slice containing label 5
        # also +3 to circumvent a lot of white space
        bottom_slice_data = label_map[:, :, bottom_slice_z]
        label_5_coordinates = np.argwhere(bottom_slice_data == 5)
        leftmost_coordinate = label_5_coordinates[np.argmax(label_5_coordinates[:, 1])]
        new_center = np.array([bottom_slice_z, leftmost_coordinate[1]+3, leftmost_coordinate[0]])
    else:
        # Find the rightmost coordinate of the point that attaches to label 7 (maximum x-coordinate)
        # also +3 to circumvent a lot of white space
        rightmost_coordinate = coordinates[np.argmin(coordinates[:, 1])]
        new_center = np.array([int(center_round[2]), rightmost_coordinate[1]+3, rightmost_coordinate[0]])

    # Create a SimpleITK image from the label map
    print(new_center)

    label_map_sitk = sitk.GetImageFromArray(label_map)
    # perform shrinking label map
    # Get the original spacing, origin, and direction
    original_spacing = label_map_sitk.GetSpacing()
    original_origin = label_map_sitk.GetOrigin()
    original_direction = label_map_sitk.GetDirection()

    # Define the scaling transformation
    transform = sitk.AffineTransform(3)
    transform.Scale([shrink_factor, shrink_factor, shrink_factor])
    
    # Define the translation to center the image on the specified coordinates
    transform.Translate(new_center * (1 - shrink_factor))

    # Apply the transformation
    label_map_shrunk_sitk = sitk.Resample(
        label_map_sitk, 
        label_map_sitk.GetSize(),
        transform,
        sitk.sitkNearestNeighbor,
        label_map_sitk.GetOrigin(),
        label_map_sitk.GetSpacing(),
        label_map_sitk.GetDirection(),
        0,
        label_map_sitk.GetPixelID()
    )
    
    # Convert back to NumPy array
    label_map_shrunk = sitk.GetArrayFromImage(label_map_shrunk_sitk)

    # ONLY select label 5 from the masks
    # Extract label 5 from the shrunk label map
    label_5_shrunk = (label_map_shrunk == 5)

    # Remove label 5 from the original label map
    label_map_no_5 = np.where(label_map == 5, 1, label_map)

    # Overlay the extracted label 5 from the shrunk label map onto the modified original label map
    result_labelmap = np.where(label_5_shrunk == 1, 5, label_map_no_5)

    # Mask for all labels except label 5 in the original label map
    non_label_5_mask = (label_map != 5) & (label_map != 1)

    # Apply the mask to the original label map and overlay these labels onto the result label map
    result_labelmap = np.where(non_label_5_mask, label_map, result_labelmap)

    # Create a new NIfTI image
    label_map_shrunk_nii = nib.Nifti1Image(result_labelmap, label_map_nii.affine, label_map_nii.header)

    return label_map_shrunk_nii

# Load the NIfTI file
input_file = f"{input_file}"
output_file = f"{outputdir}/{output_file}"
# os.makedirs(f'{outputdir}/feta_2.1_training/{map_}/anat/manipulated_maps_orig_mri_space_NOT_FINAL', exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

label_map_nii = nib.load(input_file)
shrunk_label_map_nii = shrink_label_map_nii(label_map_nii, shrink_factor)

# Save the shrunk label map
nib.save(shrunk_label_map_nii, output_file)

