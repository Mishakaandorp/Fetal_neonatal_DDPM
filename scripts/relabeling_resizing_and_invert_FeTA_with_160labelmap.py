import numpy as np
import nibabel as nib
import os
import subprocess

# This is the script to crop to the labels and resize to image dimensions.

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
    # breakpoint()
    # print(label_map.shape)
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
def map_label_values_FeTA(label_map):
    data = label_map.get_fdata()
    mapped_data = np.copy(data)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 3, 6: 3, 7: 3}
    for key, value in mapping.items():
        mapped_data[data == key] = value
    mapped_label_map = nib.Nifti1Image(mapped_data, label_map.affine)
    return mapped_label_map

# Step 0: Map the label values
def map_label_values_dHCP(label_map):
    data = label_map.get_fdata()
    mapped_data = np.copy(data)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 3, 7: 3, 8: 3, 9: 3}

    for key, value in mapping.items():
        mapped_data[data == key] = value
    mapped_label_map = nib.Nifti1Image(mapped_data, label_map.affine)
    return mapped_label_map


base_orig_dataset='dHCP_fetal_label_KISPI_good_randomvalues_fixed_20'
# FeTA_multiple
label_map_input_dir = f"../preprocessing_manipulated_labels/{base_orig_dataset}"
image_input_dir = f"../preprocessing_manipulated_labels/{base_orig_dataset}"

label_map_output_dir_160 = f"../preprocessing_manipulated_labels/{base_orig_dataset}/label_160_space/"
mri_output_dir_160 = f"../preprocessing_manipulated_labels/{base_orig_dataset}/mri_160_space/"
label_map1to7_output_dir_160 = f"../preprocessing_manipulated_labels/{base_orig_dataset}/label1to7_160_space/"

label_map_output_dir_inverted = f"../preprocessing_manipulated_labels/{base_orig_dataset}/label_original_space_inverted/"
mri_output_dir_inverted = f"../preprocessing_manipulated_labels/{base_orig_dataset}/mri_original_space_inverted/"

image_output_dir_in_between = f"../preprocessing_manipulated_labels/{base_orig_dataset}/mri_between/"
label_map_output_dir_in_between = f"../preprocessing_manipulated_labels/{base_orig_dataset}/label_between/"

orig_output_dir = f"../preprocessing_manipulated_labels/{base_orig_dataset}/orig_files/"


# Ensure output directories exist
os.makedirs(os.path.dirname(label_map_output_dir_160), exist_ok=True)
os.makedirs(os.path.dirname(mri_output_dir_160), exist_ok=True)
os.makedirs(os.path.dirname(label_map1to7_output_dir_160), exist_ok=True)

# Ensure output directories exist
os.makedirs(os.path.dirname(label_map_output_dir_inverted), exist_ok=True)
os.makedirs(os.path.dirname(mri_output_dir_inverted), exist_ok=True)

# NEW
os.makedirs(os.path.dirname(image_output_dir_in_between), exist_ok=True)
os.makedirs(os.path.dirname(label_map_output_dir_in_between), exist_ok=True)

os.makedirs(os.path.dirname(orig_output_dir), exist_ok=True)


list_label_map_input_dir = sorted(os.listdir(label_map_input_dir))
list_image_input_dir = sorted(os.listdir(image_input_dir))

# Process all files
print(len(list_label_map_input_dir))
for count, filename in enumerate(list_label_map_input_dir):
    if filename.endswith(".nii.gz"):
        print(len(list_label_map_input_dir))
        # Load the input label map and image
        input_label_file = os.path.join(label_map_input_dir, filename)
        
        # Find the base of the filename before the first '_'
        base_filename = filename.split('_')[0]
        print(base_filename)
        print(list_image_input_dir[count])
        print(list_label_map_input_dir[count])
        
        corresponding_image_files = sorted([f for f in list_image_input_dir if f.startswith(base_filename)])
        corresponding_label_files = sorted([k for k in list_label_map_input_dir if k.startswith(base_filename)])
        print(f'base_orig_dataset:{base_orig_dataset}')
        print(f'base_filename:{base_filename}')
        # breakpoint()
        
        # for i in range(len(corresponding_image_files)):
        # breakpoint()
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa')
        print(len(corresponding_image_files))
        print(len(corresponding_label_files))
        input_image_file = os.path.join(image_input_dir, list_image_input_dir[count])
        input_label_file = os.path.join(label_map_input_dir, list_label_map_input_dir[count])

        input_image_file_i = list_image_input_dir[count]
        input_label_file_i = list_label_map_input_dir[count]
        # print(input_image_file_i)
        # print(input_label_file_i)

        print(f'MRI:{input_image_file_i}')
        print(f'LABEL:{input_label_file_i}')
        # print(corresponding_label_files[i])
        # breakpoint()
        # print()
        # Load the input label map and image
        label_map = nib.load(input_label_file)
        image = nib.load(input_image_file)
        # print(image.shape)

        orig_mri_map = os.path.join(orig_output_dir, input_image_file_i)
        nib.save(image, orig_mri_map)

        orig_label_map = os.path.join(orig_output_dir, input_label_file_i)
        nib.save(label_map, orig_label_map)
        # breakpoint()
    
        # print(orig_mri_map)
        # print(orig_label_map)

        
        if base_orig_dataset == 'dHCP_neonates' or base_orig_dataset == 'dHCP_fetal_HQ_data' or base_orig_dataset == 'Korean_neonates': 
            mapped_label_map_nii = map_label_values_dHCP(label_map)
        else:
            mapped_label_map_nii = map_label_values_FeTA(label_map)
        mapped_file = os.path.join(label_map_output_dir_in_between, "mapped_labelmap.nii.gz")
        nib.save(mapped_label_map_nii, mapped_file)

        mapped_label_map=mapped_label_map_nii.get_fdata()
        # Step 1: Crop the label map
        # print(mapped_label_map.shape)
        if len(mapped_label_map.shape) > 3:
            mapped_label_map=mapped_label_map_nii.get_fdata()[:,:,:,0]
        else:
            mapped_label_map=mapped_label_map_nii.get_fdata()
        # breakpoint()
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

        cropped_image = crop_image_to_bounding_box(image, bounding_box)
        cropped_label_map_1to7 = crop_image_to_bounding_box(label_map, bounding_box)
        cropped_image_file = os.path.join(image_output_dir_in_between, "cropped_image_to_bounding_box.nii.gz")
        nib.save(cropped_image, cropped_image_file)

        # Step 2: Rescale the intensity of the cropped image
        rescaled_image, original_min, original_max = rescale_intensity_3d_based_on_labels(cropped_image,cropped_label_map)
        rescaled_file = os.path.join(image_output_dir_in_between, "rescaled_image.nii.gz")
        nib.save(rescaled_image, rescaled_file)

        size_matrix=160
        zero_matrix_size=np.zeros((size_matrix,size_matrix,size_matrix))

        # Step 3: Resize the rescaled image to 160x160x160 using c3d
        resized_rescaled_image_file = os.path.join(mri_output_dir_160, input_image_file_i)


        # label_map1to7_output_dir_160
        # print()
        # print(resized_rescaled_image_file)
        # breakpoint()
        print('KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
        resize_rescaled_command = f"c3d {rescaled_file} -interpolation Linear -resample {zero_matrix_size.shape[0]}x{zero_matrix_size.shape[1]}x{zero_matrix_size.shape[2]}vox -o '{resized_rescaled_image_file}'"
        subprocess.call(resize_rescaled_command, shell=True)

        # Step 5: Resize the cropped label map to 160x160x160 using c3d
        resized_cropped_label_file = os.path.join(label_map_output_dir_160, input_label_file_i)
        # print(resized_cropped_label_file)
        resize_cropped_command = f"c3d {cropped_file} -interpolation NearestNeighbor -resample {zero_matrix_size.shape[0]}x{zero_matrix_size.shape[1]}x{zero_matrix_size.shape[2]}vox -o '{resized_cropped_label_file}'"
        subprocess.call(resize_cropped_command, shell=True)


        resized_rescaled_1to7_labelmap_file_boundary = os.path.join(label_map_output_dir_in_between, "Labelmap_to_boundary.nii.gz")
        nib.save(cropped_label_map_1to7, resized_rescaled_1to7_labelmap_file_boundary)

        
        resized_rescaled_1to7_labelmap_file = os.path.join(label_map1to7_output_dir_160, input_label_file_i)
        # nib.save(cropped_label_map, cropped_file)
        

        print(resized_rescaled_1to7_labelmap_file)
        resize_cropped_command = f"c3d {resized_rescaled_1to7_labelmap_file_boundary} -interpolation NearestNeighbor -resample {zero_matrix_size.shape[0]}x{zero_matrix_size.shape[1]}x{zero_matrix_size.shape[2]}vox -o '{resized_rescaled_1to7_labelmap_file}'"
        subprocess.call(resize_cropped_command, shell=True)
        print('JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ')
        # breakpoint()


        # Check label sizes
        # create_matrix_
        resized_cropped_label_shape=nib.load(resized_cropped_label_file).get_fdata().shape
        print(resized_rescaled_image_file)
        resized_cropped_mri_shape=nib.load(resized_rescaled_image_file).get_fdata().shape
        print(f'resized_cropped_label_shape:{resized_cropped_label_shape}')
        print(f'resized_cropped_mri_shape:{resized_cropped_mri_shape}')
        if resized_cropped_label_shape != zero_matrix_size.shape or resized_cropped_mri_shape != zero_matrix_size.shape:
            print(f'resizing did not succeeded well. Size matrix is not {size_matrix}x{size_matrix}x{size_matrix}')


        # Step 4: Inverse resize the rescaled image back to original dimensions using c3d
        resized_back_rescaled_image_file = os.path.join(image_output_dir_in_between, "resized_back_rescaled_image.nii.gz")
        resize_back_rescaled_command = f"c3d '{resized_rescaled_image_file}' -interpolation Linear -resample {rescaled_image.shape[0]}x{rescaled_image.shape[1]}x{rescaled_image.shape[2]}vox -o '{resized_back_rescaled_image_file}'"
        subprocess.call(resize_back_rescaled_command, shell=True)

        # Step 6: Inverse resize the cropped label map back to original dimensions using c3d
        resized_back_cropped_label_file = os.path.join(label_map_output_dir_inverted, "resized_back_cropped_labelmap.nii.gz")
        resize_back_cropped_command = f"c3d '{resized_cropped_label_file}' -interpolation NearestNeighbor -resample {cropped_label_map.shape[0]}x{cropped_label_map.shape[1]}x{cropped_label_map.shape[2]}vox -o '{resized_back_cropped_label_file}'"
        subprocess.call(resize_back_cropped_command, shell=True)

        # Step 7: Inverse rescale the rescaled image
        print(resized_back_rescaled_image_file)
        img_nii_inverted_scaling=nib.load(resized_back_rescaled_image_file)
        inv_rescaled_image = inverse_rescale_intensity_3d(img_nii_inverted_scaling, original_min, original_max)
        inv_rescaled_file = os.path.join(image_output_dir_in_between, "inv_rescaled_image.nii.gz")
        nib.save(inv_rescaled_image, inv_rescaled_file)

        # Step 8: Uncrop the image
        def uncrop_image_3d(cropped_data, original_shape, bounding_box,orig_image_nifti):
            min_depth, min_row, min_col, max_depth, max_row, max_col = bounding_box
            uncropped_data = np.zeros(original_shape)
            uncropped_data[min_depth:max_depth+1, min_row:max_row+1, min_col:max_col+1] = cropped_data
            uncropped_image = nib.Nifti1Image(uncropped_data, orig_image_nifti.affine)
            return uncropped_image

        original_shape = image.shape
        label_nii_=nib.load(resized_back_cropped_label_file)

        # cropped_data = cropped_image.get_fdata()
        shape_inv_rescaled_image=inv_rescaled_image.get_fdata().shape
        shape_label_nii_=label_nii_.get_fdata().shape
        print(f'shape_inv_rescaled_image:{shape_inv_rescaled_image}')
        print(f'shape_label_nii_:{shape_label_nii_}')
        
        uncropped_image = uncrop_image_3d(inv_rescaled_image.get_fdata(), original_shape, bounding_box,image)

        uncropped_mri_file = os.path.join(mri_output_dir_inverted, input_image_file_i)
        nib.save(uncropped_image, uncropped_mri_file)

        # uncropped_label = uncrop_image_3d(label_nii_.get_fdata(), original_shape, bounding_box,image)
        # uncropped_file_label = os.path.join(label_map_output_dir_inverted, corresponding_label_files[i])
        # nib.save(uncropped_label, uncropped_file_label)

        # uncropped_label_shape=uncropped_label.shape
        uncropped_mri_shape=uncropped_image.shape
        # print(f'uncropped_label_shape:{uncropped_label_shape}')
        print(f'uncropped_mri_shape:{uncropped_mri_shape}')
        print(f'original_shape:{original_shape}')
        print()

        if uncropped_mri_shape != original_shape:
            print(f'uncropped_mri.shape is not original_shape: shape is {uncropped_mri_shape}')
            breakpoint()