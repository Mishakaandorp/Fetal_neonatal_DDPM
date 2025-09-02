import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--cortical_thickening', type=float, default=None,
                    help="Number of voxels to dilate cortex (label 3)")
parser.add_argument('--outputdir', type=str, default='')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()
map_ = args.subject
cortical_thickening = args.cortical_thickening
outputdir = args.outputdir
input_file = args.input_file
output_file = args.output_file

# Load label map
input_path = os.path.join(outputdir, input_file)
output_path = os.path.join(outputdir, output_file)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

label_map_nii = nib.load(input_path)
label_map = label_map_nii.get_fdata().astype(np.int16)

if cortical_thickening is not None and cortical_thickening > 0:
    cortex_mask = (label_map == 2)

    # Apply binary dilation with given radius (in voxels)
    dilated_cortex = binary_dilation(
        cortex_mask, 
        iterations=int(round(cortical_thickening))
    )

    # Overwrite cortex into the label map (even if other labels are present)
    label_map[dilated_cortex] = 2

# Save result
thickened_labelmap_nii = nib.Nifti1Image(label_map, label_map_nii.affine, label_map_nii.header)
nib.save(thickened_labelmap_nii, output_path)
