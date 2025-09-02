import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import subprocess

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--base_dataset', type=str, default="")
parser.add_argument('--synthesize_option', type=str, default='')

args = parser.parse_args()
base_dataset = args.base_dataset
synthesize_option = args.synthesize_option

def find_bounding_box_3d(label_map):
    non_zero_indices = np.argwhere(label_map > 0)
    if non_zero_indices.size == 0:
        return (0, 0, 0, 0, 0, 0)  # No labels found

    min_depth, min_row, min_col = non_zero_indices.min(axis=0)
    max_depth, max_row, max_col = non_zero_indices.max(axis=0)
    
    return (min_depth, min_row, min_col, max_depth, max_row, max_col)

def rescale_intensity_3d_based_on_labels(image, label_map, new_min=-1, new_max=1):
    """
    Rescale the intensity of the image based on the min and max intensity values within the labeled regions.
    Cap values higher than 1 to 1 and lower than -1 to -1.
    """
    image_data = image.get_fdata()
    label_data = label_map.get_fdata()
    
    # Mask to get the values within the labeled regions
    # breakpoint()
    labeled_voxels = image_data[label_data > 0]
    
    if labeled_voxels.size == 0:
        raise ValueError("No labeled regions found in the label map.")
    
    current_min = np.min(labeled_voxels)
    current_max = np.max(labeled_voxels)
    
    # Rescale the entire image based on the min and max values within the labeled regions
    scaled_data = (image_data - current_min) * (new_max - new_min) / (current_max - current_min) + new_min
    
    # Cap values higher than 1 to 1 and lower than -1 to -1
    scaled_data = np.clip(scaled_data, new_min, new_max)
    
    rescaled_image = nib.Nifti1Image(scaled_data, image.affine)
    
    return rescaled_image, current_min, current_max

def rescale_intensity_3d(label_map, new_min=-1, new_max=1):
    data = label_map.get_fdata()
    current_min = np.min(data)
    current_max = np.max(data)
    scaled_data = (data - current_min) * (new_max - new_min) / (current_max - current_min) + new_min
    rescaled_label_map = nib.Nifti1Image(scaled_data, label_map.affine)
    return rescaled_label_map, current_min, current_max

def inverse_rescale_intensity_3d(rescaled_label_map, original_min, original_max):
    scaled_data = rescaled_label_map.get_fdata()
    original_data = (scaled_data - np.min(scaled_data)) / (np.max(scaled_data) - np.min(scaled_data)) * (original_max - original_min) + original_min
    original_label_map = nib.Nifti1Image(original_data, rescaled_label_map.affine)
    return original_label_map

def crop_label_map_3d(label_map):
    min_depth, min_row, min_col, max_depth, max_row, max_col = find_bounding_box_3d(label_map)
    cropped_data = label_map[min_depth:max_depth+1, min_row:max_row+1, min_col:max_col+1]
    return cropped_data, (min_depth, min_row, min_col, max_depth, max_row, max_col)

def uncrop_label_map_3d(cropped_label_map, original_shape, bounding_box):
    min_depth, min_row, min_col, max_depth, max_row, max_col = bounding_box
    uncropped_data = np.zeros(original_shape)
    cropped_data = cropped_label_map.get_fdata()
    uncropped_data[min_depth:max_depth+1, min_row:max_row+1, min_col:max_col+1] = cropped_data
    uncropped_label_map = nib.Nifti1Image(uncropped_data, cropped_label_map.affine)
    return uncropped_label_map

def z_normalize_3d(label_map):
    data = label_map.get_fdata()
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    normalized_label_map = nib.Nifti1Image(normalized_data, label_map.affine)
    return normalized_label_map, mean, std

def inverse_z_normalize_3d(normalized_label_map, mean, std):
    normalized_data = normalized_label_map.get_fdata()
    original_data = normalized_data * std + mean
    original_label_map = nib.Nifti1Image(original_data, normalized_label_map.affine)
    return original_label_map

# Step 0: Map the label values
def map_label_values_example(label_map):
    data = label_map.get_fdata()
    mapped_data = np.copy(data)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 3, 6: 3, 7: 3}
    for key, value in mapping.items():
        mapped_data[data == key] = value
    mapped_label_map = nib.Nifti1Image(mapped_data, label_map.affine)
    return mapped_label_map

label_map_input_dir = f"../manipulated_labels/{base_dataset}/{synthesize_option}/all_final_labels"

label_map_output_dir_160 = f"../preprocessing_manipulated_labels/{base_dataset}/{synthesize_option}/label_160_space/"
label_map1to7_output_dir_160 = f"../preprocessing_manipulated_labels/{base_dataset}/{synthesize_option}/label1to7_160_space/"


label_map_output_dir_in_between = f"../preprocessing_manipulated_labels/{base_dataset}/{synthesize_option}/label_between/"

orig_output_dir = f"../preprocessing_manipulated_labels/{base_dataset}/{synthesize_option}/orig_files/"


# Ensure output directories exist
os.makedirs(os.path.dirname(label_map_output_dir_160), exist_ok=True)
os.makedirs(os.path.dirname(label_map1to7_output_dir_160), exist_ok=True)
os.makedirs(os.path.dirname(label_map_output_dir_in_between), exist_ok=True)
os.makedirs(os.path.dirname(orig_output_dir), exist_ok=True)


list_label_map_input_dir = sorted(os.listdir(label_map_input_dir))
# Process all files
print(f'labels to be resized: {len(list_label_map_input_dir)}')
for count, filename in enumerate(list_label_map_input_dir):
    if filename.endswith(".nii.gz"):
        input_label_file = os.path.join(label_map_input_dir, filename)
        print('')
        # Find the base of the filename before the first '_'
        base_filename = filename.split('_')[0]
        
        corresponding_label_files = sorted([k for k in list_label_map_input_dir if k.startswith(base_filename)])
        input_label_file = os.path.join(label_map_input_dir, list_label_map_input_dir[count])
        input_label_file_i = list_label_map_input_dir[count]
        print(f'resizing label: {input_label_file_i}')
        label_map = nib.load(input_label_file)
        orig_label_map = os.path.join(orig_output_dir, input_label_file_i)
        nib.save(label_map, orig_label_map)
        # else:
        mapped_label_map_nii = map_label_values_example(label_map)
        mapped_file = os.path.join(label_map_output_dir_in_between, "mapped_labelmap.nii.gz")
        nib.save(mapped_label_map_nii, mapped_file)

        mapped_label_map=mapped_label_map_nii.get_fdata()
        # Step 1: Crop the label map
        if len(mapped_label_map.shape) > 3:
            mapped_label_map=mapped_label_map_nii.get_fdata()[:,:,:,0]
        else:
            mapped_label_map=mapped_label_map_nii.get_fdata()
        cropped_data, bounding_box = crop_label_map_3d(mapped_label_map)
        cropped_label_map = nib.Nifti1Image(cropped_data, label_map.affine)

        cropped_file = os.path.join(label_map_output_dir_in_between, "cropped_labelmap.nii.gz")
        nib.save(cropped_label_map, cropped_file)

        # Step 1.1: Crop the other image using the same bounding box
        def crop_image_to_bounding_box(image, bounding_box):
            min_depth, min_row, min_col, max_depth, max_row, max_col = bounding_box
            cropped_data = image.get_fdata()[min_depth:max_depth+1, min_row:max_row+1, min_col:max_col+1]
            cropped_image = nib.Nifti1Image(cropped_data, image.affine)
            return cropped_image

        cropped_label_map_1to7 = crop_image_to_bounding_box(label_map, bounding_box)

        # Step 2: Rescale the intensity of the cropped image
        size_matrix=160
        zero_matrix_size=np.zeros((size_matrix,size_matrix,size_matrix))

        # Step 5: Resize the cropped label map to 160x160x160 using c3d
        resized_cropped_label_file = os.path.join(label_map_output_dir_160, input_label_file_i)
        resize_cropped_command = f"c3d {cropped_file} -interpolation NearestNeighbor -resample {zero_matrix_size.shape[0]}x{zero_matrix_size.shape[1]}x{zero_matrix_size.shape[2]}vox -o '{resized_cropped_label_file}'"
        subprocess.call(resize_cropped_command, shell=True)


        resized_rescaled_1to7_labelmap_file_boundary = os.path.join(label_map_output_dir_in_between, "Labelmap_to_boundary.nii.gz")
        nib.save(cropped_label_map_1to7, resized_rescaled_1to7_labelmap_file_boundary)

        
        resized_rescaled_1to7_labelmap_file = os.path.join(label_map1to7_output_dir_160, input_label_file_i)
        

        resize_cropped_command = f"c3d {resized_rescaled_1to7_labelmap_file_boundary} -interpolation NearestNeighbor -resample {zero_matrix_size.shape[0]}x{zero_matrix_size.shape[1]}x{zero_matrix_size.shape[2]}vox -o '{resized_rescaled_1to7_labelmap_file}'"
        subprocess.call(resize_cropped_command, shell=True)
        print(f' Resized label: {input_label_file_i}')