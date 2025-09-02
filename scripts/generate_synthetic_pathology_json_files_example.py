import random
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
import scipy.io
import os
import argparse
import SimpleITK as sitk
import cv2

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str)
parser.add_argument('--outputdir_json_files', type=str)
parser.add_argument('--synthesize_pathology', type=str)
parser.add_argument('--synthetic_files_total', type=int)
parser.add_argument('--outputdir', type=str, default='')


args = parser.parse_args()
subject = args.subject
sub = args.subject
outputdir_json_files = args.outputdir_json_files
synthetic_files_total = args.synthetic_files_total
synthesize_pathology = args.synthesize_pathology
print(f'synthesize_pathology {synthesize_pathology}')
outputdir = args.outputdir

# breakpoint()

import json

# File paths for 7-label map processing (new functionality)
input_file_7_labels = nib.load(f'{outputdir}/{sub}/orig_files/{sub}_7label_orig.nii.gz').get_fdata()  # Input NIfTI label map file (7-label)



def define_number_of_dilations_till_50perc(label_map_wm_half, label_map_ventricles_half, label_ventricles, label_wm):
    
    label_map_half_ventricles_sum = np.sum(label_map_ventricles_half == label_ventricles)
    label_map_half_wm_sum = np.sum(np.rint(label_map_wm_half).astype(int) == label_wm) - (np.sum(label_map_wm_half[label_map_ventricles_half==6] == label_wm))
    # breakpoint()
    percentage_half_ventricles_in_wm = (label_map_half_ventricles_sum / label_map_half_wm_sum)

    print(f'percentage_left_ventricles_in_wm 0 dilations: {percentage_half_ventricles_in_wm}')

    kernel = np.zeros((3, 3, 3), dtype=int)

    # Add a cross-like figure to the object
    kernel[1, :, 1] = 1  # Vertical line of the cross
    kernel[:, 1, 1] = 1  # Horizontal line of the cross
    kernel[1, 1, :] = 1  # Depth line of the cross
    max_dilation = 0
    # breakpoint()

    while percentage_half_ventricles_in_wm < 0.65:
        # Create a mask for the label to dilate in the specified label map
        mask = (label_map_ventricles_half == label_ventricles)

        # Perform dilation using binary dilation
        mask_new = binary_dilation(mask, kernel)
        mask_new_2 = mask_new & (label_map_ventricles_half == 3)
        # Update the label map with the dilated mask
        label_map_ventricles_half = np.where(mask_new_2, label_ventricles, label_map_ventricles_half)
        
        label_map_half_ventricles_sum = np.sum(label_map_ventricles_half == label_ventricles)
        percentage_half_ventricles_in_wm = (label_map_half_ventricles_sum / label_map_half_wm_sum)

        max_dilation += 1
        
        print(f'Dilation {max_dilation}, percentage_half_ventricles_in_wm: {percentage_half_ventricles_in_wm}')
        
    # nifti_label_map_both_halves_missing_label = nib.Nifti1Image(label_map_both_halves_missing_label, affine=ref.affine)
    print(f'max_dilation {max_dilation}')
    print(f'percentage_half_ventricles_in_wm: {percentage_half_ventricles_in_wm}')

    label_map_left_ventricles = nib.load(f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_R_ventricles.nii.gz')  # Output for modified 7-label map

    save_dilation = f'{outputdir}/{sub}/hemis_splitted/max_dilation_{max_dilation}_perc_ventr_in_wm_{percentage_half_ventricles_in_wm}.nii.gz'
    os.makedirs(os.path.dirname(save_dilation), exist_ok=True)
    nib.save(nib.Nifti1Image(label_map_ventricles_half, affine=label_map_left_ventricles.affine), save_dilation)
        
    return max_dilation


import random

def generate_next_number(possible_numbers, used_numbers):
    available_numbers = list(possible_numbers - used_numbers)

    # Remove 2 if any used number is 3, and remove 3 if any used number is 2
    if 3 in used_numbers:
        available_numbers = [x for x in available_numbers if x != 2]
    if 2 in used_numbers:
        available_numbers = [x for x in available_numbers if x != 3]

    return random.choice(available_numbers)


synthesize_pathology = synthesize_pathology.strip('"\'')

if not synthesize_pathology == 'cortical_thickness':
    label_map_left_ventricles = nib.load(f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_L_ventricles.nii.gz').get_fdata()  # Output for modified 7-label map
    label_map_right_ventricles = nib.load(f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_R_ventricles.nii.gz').get_fdata()  # Output for modified 7-label map
    modified_first_img_no_ventricles_path = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_no_ventricles.nii.gz'  # Output for modified 7-label map

    modified_first_img_no_ventricles = nib.load(modified_first_img_no_ventricles_path).get_fdata()
    label_map_left_wm = nib.load(f'{outputdir}/{sub}/hemis_splitted/{sub}_left_WM.nii.gz').get_fdata()   # Output file for first label set
    label_map_right_wm = nib.load(f'{outputdir}/{sub}/hemis_splitted/{sub}_right_WM.nii.gz').get_fdata()   # Output file for second label set
    max_dilation_left = define_number_of_dilations_till_50perc(label_map_left_wm, label_map_left_ventricles, 4, 4)
    max_dilation_right = define_number_of_dilations_till_50perc(label_map_right_wm, label_map_right_ventricles, 4, 4)

cortical_thickness_increase = 1
for file_ in range(synthetic_files_total):
    # breakpoint()
    if synthesize_pathology == 'all_pathologies':
        print(f'synthesize {synthesize_pathology}')
        
        sequence_pathologies = {}
        possible_numbers = {1, 2, 3, 4, 5}
        used_numbers = set()

        # Generate random number a
        random_number_a = generate_next_number(possible_numbers, used_numbers)
        used_numbers.add(random_number_a)
        print(f"First random number: {random_number_a}")
        # Decide with a 50% chance whether to generate a second random integer
        if random.random() < 0.5:
            random_number_b = generate_next_number(possible_numbers, used_numbers)
            used_numbers.add(random_number_b)
            print(f"Second random number: {random_number_b}")

            # Decide with a 50% chance whether to generate a third random integer
            if random.random() < 0.5:
                random_number_c = generate_next_number(possible_numbers, used_numbers)
                used_numbers.add(random_number_c)
                print(f"Third random number: {random_number_c}")
                # Decide with a 50% chance whether to generate a third random integer
                if random.random() < 0.5:
                    random_number_d = generate_next_number(possible_numbers, used_numbers)

                    print(f"Fourth random number: {random_number_d}")
                else:
                    print("No Fourth random number generated")
                    random_number_d = 0
            else:
                print("No third random number generated")
                random_number_c = 0
                random_number_d = 0
        else:
            print("No second random number generated")
            random_number_b = 0
            random_number_c = 0
            random_number_d = 0
            # random_number_c = 0

        print(f'a = {random_number_a}, b = {random_number_b}, c = {random_number_c}')
        random_value_ventriculomegaly_asymetric_L = int(random.uniform(1, max_dilation_left))
        random_value_ventriculomegaly_asymetric_R = int(random.uniform(1,max_dilation_right))

        random_value_hypo_cerebellum = random.uniform(1, 2)
        random_value_hypo_pontocerebellar = random.uniform(0.5, 1)
        random_value_microcephaly = random.uniform(1, 1.1)

        sequence_pathologies['subject'] = subject
        random_value_pnh = int(random.uniform(1, 50))

        if random_number_a == 1 or random_number_b == 1 or random_number_c == 1:
            if random.random() < 0.5:

                sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
            else:
                sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
            # sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
            # sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['ventriculomegaly_asymetric_L'] = 0
            sequence_pathologies['ventriculomegaly_asymetric_R'] = 0
        if random_number_a == 2 or random_number_b == 2 or random_number_c == 2:
            sequence_pathologies['hypo_cerebellum'] = random_value_hypo_cerebellum
        else:
            sequence_pathologies['hypo_cerebellum'] = 1
        if random_number_a == 3 or random_number_b == 3 or random_number_c == 3:
            sequence_pathologies['hypo_pontocerebellar'] = random_value_hypo_pontocerebellar
        else:
            sequence_pathologies['hypo_pontocerebellar'] = 1
        if random_number_a == 4 or random_number_b == 4 or random_number_c == 4:
            sequence_pathologies['microcephaly'] = random_value_microcephaly
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['microcephaly'] = 1
        if random_number_a == 5 or random_number_b == 5 or random_number_c == 5 or random_number_d == 5:
            sequence_pathologies['pnh'] = random_value_pnh
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['pnh'] = 0
    elif synthesize_pathology == 'noPNH':
        print(f'synthesize {synthesize_pathology}')

        sequence_pathologies = {}
        possible_numbers = {1, 2, 3, 4}
        used_numbers = set()

        # Generate random number a
        random_number_a = generate_next_number(possible_numbers, used_numbers)
        used_numbers.add(random_number_a)
        print(f"First random number: {random_number_a}")
        # Decide with a 50% chance whether to generate a second random integer
        if random.random() < 0.5:
            random_number_b = generate_next_number(possible_numbers, used_numbers)
            used_numbers.add(random_number_b)
            print(f"Second random number: {random_number_b}")

            # Decide with a 50% chance whether to generate a third random integer
            if random.random() < 0.5:
                random_number_c = generate_next_number(possible_numbers, used_numbers)
                used_numbers.add(random_number_c)
            
                random_number_d = 10
            else:
                print("No third random number generated")
                random_number_c = 0
                random_number_d = 0
        else:
            print("No second random number generated")
            random_number_b = 0
            random_number_c = 0
            random_number_d = 0
            # random_number_c = 0

        print(f'a = {random_number_a}, b = {random_number_b}, c = {random_number_c}')
        random_value_ventriculomegaly_asymetric_L = int(random.uniform(1, max_dilation_left))
        random_value_ventriculomegaly_asymetric_R = int(random.uniform(1,max_dilation_right))

        random_value_hypo_cerebellum = random.uniform(1, 2)
        random_value_hypo_pontocerebellar = random.uniform(0.5, 1)
        random_value_microcephaly = random.uniform(1, 1.1)

        sequence_pathologies['subject'] = subject
        random_value_pnh = int(random.uniform(1, 50))

        if random_number_a == 1 or random_number_b == 1 or random_number_c == 1:
            if random.random() < 0.5:

                sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
            else:
                sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['ventriculomegaly_asymetric_L'] = 0
            sequence_pathologies['ventriculomegaly_asymetric_R'] = 0
        if random_number_a == 2 or random_number_b == 2 or random_number_c == 2:
            sequence_pathologies['hypo_cerebellum'] = random_value_hypo_cerebellum
        else:
            sequence_pathologies['hypo_cerebellum'] = 1
        if random_number_a == 3 or random_number_b == 3 or random_number_c == 3:
            sequence_pathologies['hypo_pontocerebellar'] = random_value_hypo_pontocerebellar
        else:
            sequence_pathologies['hypo_pontocerebellar'] = 1
        if random_number_a == 4 or random_number_b == 4 or random_number_c == 4:
            sequence_pathologies['microcephaly'] = random_value_microcephaly
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['microcephaly'] = 1
        if random_number_a == 5 or random_number_b == 5 or random_number_c == 5 or random_number_d == 5:
            sequence_pathologies['pnh'] = random_value_pnh
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['pnh'] = 0
    elif synthesize_pathology == 'ventriculomegaly':
        print(f'synthesize {synthesize_pathology}')

        # available_numbers = list(possible_numbers - used_numbers)

        sequence_pathologies = {}
        possible_numbers = {1}
        used_numbers = set()

        # Generate random number a
        random_number_a = generate_next_number(possible_numbers, used_numbers)
        used_numbers.add(random_number_a)
        print(f"First random number: {random_number_a}")
        
        random_number_b = 10
        random_number_c = 10
        random_number_d = 10
        # random
        print(f'a = {random_number_a}')
        random_value_ventriculomegaly_asymetric_L = int(random.uniform(1, max_dilation_left))
        random_value_ventriculomegaly_asymetric_R = int(random.uniform(1,max_dilation_right))

        random_value_hypo_cerebellum = random.uniform(1, 2)
        random_value_hypo_pontocerebellar = random.uniform(0.5, 1)
        random_value_microcephaly = random.uniform(1, 1.1)

        sequence_pathologies['subject'] = subject
        random_value_pnh = int(random.uniform(1, 30))
        if random.random() < 0.5:

            sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
            sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
        else:
            sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
            sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        
        if random_number_a == 2 or random_number_b == 2 or random_number_c == 2:
            sequence_pathologies['hypo_cerebellum'] = random_value_hypo_cerebellum
        else:
            sequence_pathologies['hypo_cerebellum'] = 1
        if random_number_a == 3 or random_number_b == 3 or random_number_c == 3:
            sequence_pathologies['hypo_pontocerebellar'] = random_value_hypo_pontocerebellar
        else:
            sequence_pathologies['hypo_pontocerebellar'] = 1
        if random_number_a == 4 or random_number_b == 4 or random_number_c == 4:
            sequence_pathologies['microcephaly'] = random_value_microcephaly
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['microcephaly'] = 1
        if random_number_a == 5 or random_number_b == 5 or random_number_c == 5 or random_number_d == 5:
            sequence_pathologies['pnh'] = random_value_pnh
            if sequence_pathologies['ventriculomegaly_asymetric_L'] == 0 and sequence_pathologies['ventriculomegaly_asymetric_R'] == 0:
                if random.random() < 0.5:

                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = sequence_pathologies['ventriculomegaly_asymetric_L']
                else:
                    sequence_pathologies['ventriculomegaly_asymetric_L'] = random_value_ventriculomegaly_asymetric_L
                    sequence_pathologies['ventriculomegaly_asymetric_R'] = random_value_ventriculomegaly_asymetric_R
        else:
            sequence_pathologies['pnh'] = 0
    elif synthesize_pathology == 'cortical_thickness':
        print(f'synthesize {synthesize_pathology}')
        sequence_pathologies = {}
        
        sequence_pathologies['subject'] = subject
        sequence_pathologies['cortical_thickness'] = cortical_thickness_increase

        # No ventriculomegaly, no other pathologies
        sequence_pathologies['ventriculomegaly_asymetric_L'] = 0
        sequence_pathologies['ventriculomegaly_asymetric_R'] = 0
        sequence_pathologies['hypo_cerebellum'] = 1
        sequence_pathologies['hypo_pontocerebellar'] = 1
        sequence_pathologies['microcephaly'] = 1
        sequence_pathologies['pnh'] = 0
        sequence_pathologies['max_dilation_left'] = 0
        sequence_pathologies['max_dilation_right'] = 0
        cortical_thickness_increase +=1
    
    sequence_pathologies['synthetic_file'] = file_
    if not synthesize_pathology == 'cortical_thickness':
        sequence_pathologies['max_dilation_left'] = max_dilation_left
        sequence_pathologies['max_dilation_right'] = max_dilation_right
        sequence_pathologies['cortical_thickness'] = 0
    sequence_pathologies['synthetic_files_total'] = synthetic_files_total
    print(sequence_pathologies)
    import json

    file_path = f'{outputdir_json_files}/{subject}_json_{file_}.json'
    print(f'file_path:{file_path}')
    print(os.path.dirname(file_path))
    # breakpoint()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(sequence_pathologies, json_file, indent=4)
    print(file_path)
    print(f"JSON data has been written")