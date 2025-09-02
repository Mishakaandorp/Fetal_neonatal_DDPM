import nibabel as nib
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="", help="Subject identifier")
parser.add_argument('--outputdir', type=str, default='', help="Output directory")
parser.add_argument('--input_file', type=str, required=True, help="Input file name")
parser.add_argument('--orig_file', type=str, required=True, help="Original file name")
parser.add_argument('--output_file', type=str, required=True, help="Output file name")

args = parser.parse_args()

subject = args.subject
outputdir = args.outputdir
input_file = args.input_file
orig_file = args.orig_file
output_file = args.output_file

# Function to substitute label map values
def substitute_labelmap(orig_label_map, modified_label_map):
    mask = (orig_label_map != 5) & (orig_label_map != 1) & (orig_label_map != 7) & (orig_label_map != 4)
    modified_label_map[mask] = orig_label_map[mask]
    return modified_label_map

# Load input and original files
input_path = os.path.join(outputdir, input_file)
orig_path = os.path.join(outputdir, orig_file)
# breakpoint()
input_nii = nib.load(input_path)
input_label_map = input_nii.get_fdata()

orig_nii = nib.load(orig_path)
orig_label_map = orig_nii.get_fdata()

# Substitute label map values
modified_labelmap = substitute_labelmap(orig_label_map, input_label_map)

# Save the modified label map
output_path = os.path.join(outputdir, output_file)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
nib.save(nib.Nifti1Image(modified_labelmap, orig_nii.affine), output_path)
