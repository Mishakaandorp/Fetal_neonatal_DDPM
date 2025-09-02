import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
import scipy.io
import os
import argparse
import SimpleITK as sitk

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--Ventriculomegaly_R', type=int)
parser.add_argument('--Ventriculomegaly_L', type=int)
parser.add_argument('--input_file_label', type=str)
parser.add_argument('--output_label', type=str)
parser.add_argument('--outputdir', type=str, default='')
args = parser.parse_args()

# Assign parsed arguments to variables
sub = args.subject
Ventriculomegaly_L = args.Ventriculomegaly_L
Ventriculomegaly_R = args.Ventriculomegaly_R
input_file_label = args.input_file_label
output_label = args.output_label
outputdir = args.outputdir



def dilate_specific_label_in_specific_map(label_map, label_to_dilate, label_dilate_in, dilation_iterations, kernel):
    for _ in range(dilation_iterations):
        # Create a mask for the label to dilate in the specified label map
        mask = (label_map == label_to_dilate)

        # Perform dilation using binary dilation
        mask_new = binary_dilation(mask, kernel)
        mask_new_2 = mask_new & (label_map == label_dilate_in)

        # Update the label map with the dilated mask
        label_map = np.where(mask_new_2, label_to_dilate, label_map)

    return label_map


def dilate_specific_label_in_specific_map_GM(label_map, label_to_dilate, label_dilate_in, dilation_iterations, kernel):
    for _ in range(dilation_iterations):
        # Create a mask for the label to dilate in the specified label map

        mask = np.isin(label_map, [0,1,2])
        # breakpoint()
        mask_new = binary_dilation(mask, kernel)
        mask_new_2 = mask_new & (label_map == label_dilate_in)

        # Update the label map with the dilated mask
        label_map = np.where(mask_new_2, label_to_dilate, label_map)

    return label_map

def erode_specific_label_in_specific_map(label_map, label_to_erode, label_erode_in, erosion_iterations, kernel):
    for _ in range(erosion_iterations):
        # Create a mask for the label to erode in the specified label map
        mask = (label_map == label_to_erode)

        # Perform erosion using binary erosion
        mask_new = binary_erosion(mask, kernel)

        # Determine the regions to be overwritten by label_erode_in
        mask_new_2 = mask & ~mask_new

        # Update the label map with the eroded mask
        label_map = np.where(mask_new_2, label_erode_in, label_map)

    return label_map

def gaussian_smoothing(label_map, label_map_to_smooth, label_to_smooth, label_to_not_smooth_in, label_dilate_in):
    # Apply gaussian smoothing
    ventrikels_only_mask = np.zeros(label_map.shape)
    ventrikels_only_mask[label_map_to_smooth == label_to_smooth] = 1
    gaus_ventrkels_smooth5 = scipy.ndimage.gaussian_filter(ventrikels_only_mask, 5)
    gaus_ventrkels_smooth5_03 = np.copy(label_map_to_smooth)
    ventrikels_dilated = (gaus_ventrkels_smooth5 > 0.3)
    gaus_ventrkels_smooth5_03[ventrikels_dilated] = label_to_smooth
    
    # Overwrite ventrikel map with LABEL 
    for label in range(round(label_map.max())+1):
        if label != label_to_smooth and label != label_dilate_in:
            # print(label)
            labels_dilated = (label_map_to_smooth == label)
            gaus_ventrkels_smooth5_03[labels_dilated] = label

    # Extract ventrikels and use the overwrited ventrikel map on the original label map 
    labels_ventrikels = (gaus_ventrkels_smooth5_03 == label_to_smooth)
    label_map_to_smooth_dlilated_ventrikels_gaussian = np.copy(label_map_to_smooth)
    label_map_to_smooth_dlilated_ventrikels_gaussian[labels_ventrikels] = label_to_smooth

    return label_map_to_smooth_dlilated_ventrikels_gaussian

# Potentially do for eroded ventricles
def shrink_label_map_nii(label_map_nii, output_file_2,shrink_factor=0.5):
    # Load the label map data
    label_map = label_map_nii.get_fdata()
    
    # Ensure the label map has values between 0 and the number of labels
    unique_labels = np.unique(label_map)
    max_label = unique_labels.max()
    
    # Create a mask of all non-zero labels (assuming 0 is the background)
    mask = (label_map > 0).astype(np.uint8)
    
    # Find the center of the masked region
    indices = np.argwhere(mask)
    if len(indices) > 0:
        center = indices.mean(axis=0)
    else:
        center = np.array(label_map.shape) / 2

    # Create a SimpleITK image from the label map
    label_map_sitk = sitk.GetImageFromArray(label_map)
    
    # Get the original spacing, origin, and direction
    original_spacing = label_map_sitk.GetSpacing()
    original_origin = label_map_sitk.GetOrigin()
    original_direction = label_map_sitk.GetDirection()

    # Define the scaling transformation
    transform = sitk.AffineTransform(3)
    transform.Scale([shrink_factor, shrink_factor, shrink_factor])
    
    # Define the translation to center the image on the center of the mask
    transform.Translate(np.array(center) * (1 - shrink_factor))

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
        
    label_map_shrunk_nii = nib.Nifti1Image(label_map_shrunk, label_map_nii.affine, label_map_nii.header)

    return label_map_shrunk_nii

label_to_dilate_GM = 2
label_to_dilate_ventrikels = 4
label_cerebellum = 5
label_brainfluid = 1
label_dilate_in = 3

label_erode_with = 3
label_to_erode_ventrikels = 4

kernel = np.zeros((3, 3, 3), dtype=int)

# Add a cross-like figure to the object
kernel[1, :, 1] = 1  # Vertical line of the cross
kernel[:, 1, 1] = 1  # Horizontal line of the cross
kernel[1, 1, :] = 1  # Depth line of the cross

# Main logic
label_to_dilate_GM = 2
label_to_dilate_ventrikels = 4
label_cerebellum = 5
label_brainfluid = 1
label_dilate_in = 3

label_erode_with = 3
label_to_erode_ventrikels = 4

# hardcoded dummy
# Determine whether to perform erosion or dilation based on the Ventriculomegaly values
if Ventriculomegaly_L < 0 or Ventriculomegaly_R < 0:
    Ventriculomegaly_L = np.abs(Ventriculomegaly_L)
    Ventriculomegaly_R = np.abs(Ventriculomegaly_R)
    do_erosion = True
    do_dilation = False
else:
    do_erosion = False
    do_dilation = True

if do_dilation:
    print(f'Doing dilation: Ventriculomegaly_L {Ventriculomegaly_L}, Ventriculomegaly_R {Ventriculomegaly_R}')
    map_subj_segm = f"{outputdir}/{input_file_label}"
    ref = nib.load(map_subj_segm)
    label_map_ref = ref.get_fdata()
    label_map = label_map_ref[:, :, :]


    # Split label_maps from another script
    # label_map_orig = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/template_registrations_Misha_final/feta_2.1_training/{sub}/anat/orig_maps/{sub}_seg_orig.nii.gz'
    # label_map_orig_shrinked = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/template_registrations_Misha_final/feta_2.1_training/{sub}/anat/orig_maps/{sub}_seg_orig_shrinked_0.6.nii.gz'

    label_map_first_half_file = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_L_ventricles.nii.gz'
    label_map_second_half_file = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_R_ventricles.nii.gz'
    modified_first_img_no_ventricles_path = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_no_ventricles.nii.gz'

    label_map_first_half = nib.load(label_map_first_half_file).get_fdata()
    label_map_second_half = nib.load(label_map_second_half_file).get_fdata()
    label_map_first_half_nib = nib.load(label_map_first_half_file)
    
    modified_first_img_no_ventricles = nib.load(modified_first_img_no_ventricles_path).get_fdata()
    
    final_map = np.copy(modified_first_img_no_ventricles)

    pixels_in_between_GM_ventrikels = 2
    if Ventriculomegaly_L > 0:
        print(f'Dilating left ventriculomegaly with {Ventriculomegaly_L} iterations.')
        dilated_label_map_GM_first_half = dilate_specific_label_in_specific_map_GM(label_map_first_half, label_to_dilate_GM, label_dilate_in, pixels_in_between_GM_ventrikels, kernel)
        
        save_dilation = f"{outputdir}/{sub}/ventriculomegaly/{sub}_dilated_label_map_GM_first_half_.nii.gz"
        os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
        nib.save(nib.Nifti1Image(dilated_label_map_GM_first_half, affine=label_map_first_half_nib.affine), save_dilation)
        
        dilated_label_map_ventrikels_first_half = dilate_specific_label_in_specific_map(dilated_label_map_GM_first_half, label_to_dilate_ventrikels, label_dilate_in, Ventriculomegaly_L, kernel)
        
        save_dilation = f"{outputdir}/{sub}/ventriculomegaly/{sub}_dilated_label_map_ventrikels_first_half.nii.gz"
        os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
        nib.save(nib.Nifti1Image(dilated_label_map_ventrikels_first_half, affine=label_map_first_half_nib.affine), save_dilation)

        # save_dilation = f"{outputdir}/{sub}/ventriculomegaly/{sub}_dilated_label_map_ventrikels_first_half.nii.gz"
        # os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
        # nib.save(nib.Nifti1Image(dilated_label_map_GM_first_half, affine=label_map_first_half_nib.affine), save_dilation)
        
        smoothed_label_map_first_half_and_GM = gaussian_smoothing(label_map, dilated_label_map_ventrikels_first_half, label_to_dilate_ventrikels, label_to_dilate_GM, label_dilate_in)
        
        save_dilation = f"{outputdir}/{sub}/ventriculomegaly/{sub}_smoothed_label_map_first_half_and_GM.nii.gz"
        os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
        nib.save(nib.Nifti1Image(smoothed_label_map_first_half_and_GM, affine=label_map_first_half_nib.affine), save_dilation)

        labels_ventrikels = (smoothed_label_map_first_half_and_GM == label_to_dilate_ventrikels)
        
    else:
        # Extract ventrikels and use the overwrited ventrikel map on the original label map 
        labels_ventrikels = (label_map_first_half == label_to_dilate_ventrikels)
    # labels_ventrikels = (label_map_first_half == label_to_dilate_ventrikels)
    
    final_map[labels_ventrikels] = label_to_dilate_ventrikels

    if Ventriculomegaly_R > 0:
        print(f'Dilating right ventriculomegaly with {Ventriculomegaly_R} iterations.')

        dilated_label_map_GM_second_half = dilate_specific_label_in_specific_map_GM(label_map_second_half, label_to_dilate_GM, label_dilate_in, pixels_in_between_GM_ventrikels, kernel)
        
        dilated_label_map_ventrikels_second_half = dilate_specific_label_in_specific_map(dilated_label_map_GM_second_half, label_to_dilate_ventrikels, label_dilate_in, Ventriculomegaly_R, kernel)

        save_dilation = f"{outputdir}/{sub}/ventriculomegaly/{sub}_dilated_label_map_ventrikels_second_half.nii.gz"
        os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
        nib.save(nib.Nifti1Image(dilated_label_map_ventrikels_second_half, affine=label_map_first_half_nib.affine), save_dilation)

        smoothed_label_map_second_half_and_GM = gaussian_smoothing(label_map, dilated_label_map_ventrikels_second_half, label_to_dilate_ventrikels, label_to_dilate_GM, label_dilate_in)
        
        labels_ventrikels = (smoothed_label_map_second_half_and_GM == label_to_dilate_ventrikels)
        
    else:
        # Extract ventrikels and use the overwrited ventrikel map on the original label map 
        labels_ventrikels = (label_map_second_half == label_to_dilate_ventrikels)
    
    labels_ventrikels_orig= (label_map==label_to_dilate_ventrikels)

    final_map[labels_ventrikels] = label_to_dilate_ventrikels
    final_map[labels_ventrikels_orig] = label_to_dilate_ventrikels
            
    output_file_ = f"{outputdir}/{output_label}"
    print(f'Saving file .....')

    os.makedirs(os.path.dirname(output_file_), exist_ok=True)
    # map_subj_segm = f"{outputdir}/{input_file_label}"

    nib.save(nib.Nifti1Image(final_map, affine=ref.affine), output_file_)

    print(f'saved dilation to: {output_file_}')
    #breakpoint()

# Potential erosion
# Results in seperate uncontinous islands. 
# Potentially do with shrinking brain...
if do_erosion:
            print(f'do_erosion Ventriculomegaly_L {Ventriculomegaly_L} Ventriculomegaly_R {Ventriculomegaly_R}')
            map_subj_segm = f"{outputdir}/{input_file_label}"
            ref = nib.load(map_subj_segm)
            label_map_ref = ref.get_fdata()
            label_map = label_map_ref[:,:,:]

            # Split label_maps
            label_map_orig = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/template_registrations_Misha_final/feta_2.1_training/{sub}/anat/orig_maps/{sub}_seg_orig.nii.gz'
            label_map_orig_shrinked = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/template_registrations_Misha_final/feta_2.1_training/{sub}/anat/orig_maps/{sub}_seg_orig_shrinked_0.6.nii.gz'
            label_map_first_half_file = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/split_hemispheres/venticles_splited/{sub}_ventricles_left_derotated_orig.nii.gz'
            label_map_second_half_file = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/split_hemispheres/venticles_splited/{sub}_ventricles_right_derotated_orig.nii.gz'
            modified_first_img_no_ventricles_path = f'/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/feta_aligned_to_template/split_hemispheres/venticles_splited/{sub}_both_ventricles_missing.nii.gz'


            label_map_first_half = nib.load(label_map_first_half_file).get_fdata()
            label_map_second_half = nib.load(label_map_second_half_file).get_fdata()
            label_map_first_half_nib = nib.load(label_map_first_half_file)
            # breakpoint()
            label_map_orig_nib = nib.load(label_map_orig)
            # label_map_orig_shrinked_nib = nib.load(label_map_orig_shrinked)
            shrinked_label = shrink_label_map_nii(label_map_orig_nib, label_map_orig_shrinked,shrink_factor=1.3)
            nib.save(shrinked_label, label_map_orig_shrinked)
            
            label_map_first_half = nib.load(label_map_first_half_file).get_fdata()
            label_map_second_half = nib.load(label_map_second_half_file).get_fdata()
            modified_first_img_no_ventricles = nib.load(modified_first_img_no_ventricles_path).get_fdata()
            
            # label_map_first_half, label_map_second_half, label_map_both_halves_missing_label = seperating_labels_halfs(label_map, label_to_dilate_ventrikels,shift_z, label_dilate_in)
            ## Apply dilation to ventrikels first half hemisphere
            final_map = np.copy(modified_first_img_no_ventricles)
            
            if Ventriculomegaly_L > 0:               
                eroded_label_map_ventrikels_first_half = erode_specific_label_in_specific_map(label_map_first_half, label_to_erode_ventrikels, label_erode_with, Ventriculomegaly_L, kernel)

                smoothed_label_map_first_half = gaussian_smoothing(label_map, eroded_label_map_ventrikels_first_half, label_to_erode_ventrikels, label_to_dilate_GM, label_erode_with)
                
                labels_ventrikels = (smoothed_label_map_first_half == label_to_erode_ventrikels)
                
            else:
                # Extract ventrikels and use the overwrited ventrikel map on the original label map 
                labels_ventrikels = (label_map_first_half == label_to_erode_ventrikels)
            
            final_map[labels_ventrikels] = label_to_erode_ventrikels

            if Ventriculomegaly_R > 0:               
                eroded_label_map_ventrikels_second_half = erode_specific_label_in_specific_map(label_map_second_half, label_to_erode_ventrikels, label_erode_with, Ventriculomegaly_R, kernel)

                smoothed_label_map_second_half = gaussian_smoothing(label_map, eroded_label_map_ventrikels_second_half, label_to_erode_ventrikels, label_to_dilate_GM, label_erode_with)
                
                labels_ventrikels = (smoothed_label_map_second_half == label_to_erode_ventrikels)
                
            else:
                # Extract ventrikels and use the overwrited ventrikel map on the original label map 
                labels_ventrikels = (label_map_second_half == label_to_erode_ventrikels)
            
            final_map[labels_ventrikels] = label_to_erode_ventrikels
            
            
            output_file_ = f"{outputdir}/{output_label}"
            print(f'Saving file .....')

            os.makedirs(os.path.dirname(output_file_), exist_ok=True)
            # map_subj_segm = f"{outputdir}/{input_file_label}"

            nib.save(nib.Nifti1Image(final_map, affine=ref.affine), output_file_)
            # print('saved dilation')
            print(f'saved dilation to: {output_file_}')

            