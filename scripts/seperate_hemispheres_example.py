import sys as sys
import subprocess
import tempfile

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
import trimesh

from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian
from sklearn.cluster import BisectingKMeans

import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="")
parser.add_argument('--outputdir', type=str, default='')
args = parser.parse_args()

# Assign parsed arguments to variables
sub = args.subject
outputdir = args.outputdir


# from CortexODE_main.util.tca import topology

def extract_ventricles_feta(path_seg, path_wm):
    """Generate a whole-brain white matter mask from FETA tissue segmentation

    The binary white matter mask includes the following FETA tissues:
      + white matter (WM)
      + ventricles (lateral and 4th ventricles)
      + deep grey matter

    Similarly to what is done is most software such as FreeSurfer,
    these tissues are added to ease brain hemisphere splitting and
    topologically correct surface from per hemisphere white matter mask

    TODO: remove 4th ventricle as it influences hemisphere splitting

    Parameters
    ----------
    path_seg: str
            Path of the whole brain FETA tissue segmentation volume
    path_wm: str
            Path of the whole white matter binary mask volume

    Returns
    -------
    """
    vol = nib.load(path_seg)
    data = vol.get_fdata()
    new_data = data.copy()
    # insure integer values
    data = np.round(data)
    data = data.astype(np.uint16)

    new_data[data == 1] = 0  # cerebro-spinal fluid
    new_data[data == 2] = 0  # cortical grey matter
    new_data[data == 5] = 0  # brainstem
    new_data[data == 7] = 0  # 
    new_data[data == 6] = 0  # deep grey matter
    new_data[data == 3] = 0  # white matter
    new_data[data == 4] = 1  # ventricles

    new_data = new_data.astype(np.uint16)
    wm_vol = nib.Nifti1Image(new_data, affine=vol.affine)
    nib.save(wm_vol, path_wm)

def extract_wm_feta(path_seg, path_wm):
    """Generate a whole-brain white matter mask from FETA tissue segmentation

    The binary white matter mask includes the following FETA tissues:
      + white matter (WM)
      + ventricles (lateral and 4th ventricles)
      + deep grey matter

    Similarly to what is done is most software such as FreeSurfer,
    these tissues are added to ease brain hemisphere splitting and
    topologically correct surface from per hemisphere white matter mask

    TODO: remove 4th ventricle as it influences hemisphere splitting

    Parameters
    ----------
    path_seg: str
            Path of the whole brain FETA tissue segmentation volume
    path_wm: str
            Path of the whole white matter binary mask volume

    Returns
    -------
    """
    vol = nib.load(path_seg)
    data = vol.get_fdata()
    new_data = data.copy()
    # insure integer values
    data = np.round(data)
    data = data.astype(np.uint16)

    new_data[data == 1] = 0  # cerebro-spinal fluid
    new_data[data == 2] = 0  # cortical grey matter
    new_data[data == 5] = 0  # brainstem
    new_data[data == 7] = 0  # 
    new_data[data == 6] = 1  # deep grey matter
    new_data[data == 3] = 1  # white matter
    new_data[data == 4] = 1  # ventricles

    new_data = new_data.astype(np.uint16)
    wm_vol = nib.Nifti1Image(new_data, affine=vol.affine)
    nib.save(wm_vol, path_wm)

def find_and_save_overlap(file1, label1, file2, label2, output_file):
    """
    Find the overlapping regions between two label maps based on specified labels and save to a new NIfTI file.
    
    Parameters:
    file1 (str): File path to the first label map (NIfTI format).
    label1 (int): Label to isolate in the first label map.
    file2 (str): File path to the second label map (NIfTI format).
    label2 (int): Label to isolate in the second label map.
    output_file (str): File path to save the resulting overlap label map (NIfTI format).
    """
    # Load the label maps using nibabel
    img1 = nib.load(file1)
    img2 = nib.load(file2)

    label_map1 = img1.get_fdata()
    label_map2 = img2.get_fdata()

    # Create binary masks for the specified labels
    mask1 = (label_map1 == label1)
    mask2 = (label_map2 == label2)

    # Find overlapping regions
    overlap_mask = mask1 & mask2

    # Create a new label map with only the overlapping regions
    overlap_label_map = np.zeros_like(label_map1)
    overlap_label_map[overlap_mask] = label_map1[overlap_mask]

    # Save the resulting overlap label map to a new NIfTI file
    overlap_img = nib.Nifti1Image(overlap_label_map, img1.affine, img1.header)
    nib.save(overlap_img, output_file)

    print(f"Overlap label map saved to {output_file}")




def extract_gm_dHCPlike(path_seg, path_wm):
    """Generate a whole-brain white matter mask from dHCP like tissue segmentation

    The binary white matter mask includes the following dHCP tissues:
      + white matter (WM)
      + ventricles (lateral and 4th ventricles)
      + deep grey matter
      + Amygdala

    Similarly to what is done is most software such as FreeSurfer,
    these tissues are added to ease brain hemisphere splitting and
    topologically correct surface from per hemisphere white matter mask

    Parameters
    ----------
    path_seg: str
            Path of the whole brain dHCPlike tissue segmentation volume
    path_wm: str
            Path of the whole white matter binary mask volume

    Returns
    -------
    """
    vol = nib.load(path_seg)
    data = vol.get_fdata()
    new_data = data.copy()
    # insure integer values
    data = np.round(data)
    data = data.astype(np.uint16)

    new_data[data == 1] = 0  # cerebro-spinal fluid
    new_data[data == 2] = 1  # cortical grey matter
    new_data[data == 5] = 1  # cerebellum
    new_data[data == 7] = 1  # brainstem
    new_data[data == 6] = 0  # deep grey matter
    new_data[data == 3] = 1  # white matter
    new_data[data == 4] = 0  # ventricles
    new_data[data == 8] = 0  # Amygdala 
    new_data[data == 9] = 1  # Amygdala 
     
    new_data = new_data.astype(np.uint16)
    wm_vol = nib.Nifti1Image(new_data, affine=vol.affine)
    nib.save(wm_vol, path_wm)


def get_hemisphere_masks(
    path_brain_mask, path_left_hemisphere_mask, path_right_hemisphere_mask
):
    """Split a full brain binary mask into two masks, one per hemisphere

    Parameters
    ----------
    path_brain_mask : str
                     Path of the full brain binary mask volume
    path_left_hemisphere_mask: str
                     Path of the left hemisphere binary mask volume
    path_right_hemisphere_mask:
                     Path of the right hemisphere binary mask volume
    Returns
    -------
    """

    vol = nib.load(path_brain_mask)

    # Insure Voxel Space is in RAS+ convention
    img_ornt = io_orientation(vol.affine)
    ras_ornt = axcodes2ornt("RAS")
    to_canonical = ornt_transform(img_ornt, ras_ornt)
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    vol_canonical = vol.as_reoriented(to_canonical)

    data = vol_canonical.get_fdata()
    # Consider mask voxels only
    indexes = np.where(data != 0)
    # Keep only L-R and I-S axis (i.e. clustering in the coronal plane)
    vox_coord = np.zeros((len(indexes[0]), 2), dtype=np.int16)
    vox_coord[:, 0] = indexes[0]
    vox_coord[:, 1] = indexes[2]
    km = BisectingKMeans(n_clusters=2, n_init=10, init="k-means++")
    km.fit(vox_coord)
    clusters = km.labels_.copy()
    labels = np.zeros(data.shape, dtype=np.uint8)
    centroids = km.cluster_centers_
    # In RAS+ convention Right > Left
    # 127 --> left Hemisphere
    # 255 --> right hemisphere
    if centroids[0][0] > centroids[1][0]:
        clusters[clusters == 0] = 255
        clusters[clusters == 1] = 127
    else:
        clusters[clusters == 0] = 127
        clusters[clusters == 1] = 255
    labels[indexes] = clusters

    # generate left hemisphere mask
    left_seg = np.zeros(labels.shape, dtype=np.uint8)
    left_seg[labels == 127] = 1
    left_vol = nib.Nifti1Image(left_seg, vol_canonical.affine)
    left_vol = left_vol.as_reoriented(from_canonical)
    nib.save(left_vol, path_left_hemisphere_mask)

    # generate right hemisphere mask
    right_seg = np.zeros(labels.shape, dtype=np.uint8)
    right_seg[labels == 255] = 1
    right_vol = nib.Nifti1Image(right_seg, vol_canonical.affine)
    right_vol = right_vol.as_reoriented(from_canonical)
    right_vol.header.affine = vol.affine
    nib.save(right_vol, path_right_hemisphere_mask)

    return labels

def modify_label_map(first_label_map_path, second_label_map_path, modified_label_map_path,modified_first_img_no_ventricles_path):
    """
    Modify the first label map by changing label 4 to 3, then using regions 
    labeled 1 in the second label map to set corresponding regions in the 
    first label map to 4.

    Parameters:
    - first_label_map_path: str, path to the first label map NIfTI file.
    - second_label_map_path: str, path to the second label map NIfTI file.
    - modified_label_map_path: str, path to save the modified first label map.
    """

    # Load the first label map
    first_img = nib.load(first_label_map_path)
    first_data = first_img.get_fdata()

    # Change label 4 to 3 in the first label map
    first_data[first_data == 4] = 3

    # Load the second label map
    second_img = nib.load(second_label_map_path)
    second_data = second_img.get_fdata()
    
    # Save the modified first label map
    modified_first_img_no_ventricles = nib.Nifti1Image(first_data, first_img.affine, first_img.header)
    nib.save(modified_first_img_no_ventricles, modified_first_img_no_ventricles_path)
    # Find regions in the second label map with label 1
    regions_to_change = (second_data == 1)

    # Set those regions in the first label map to 4
    first_data[regions_to_change] = 4

    # Save the modified first label map
    modified_first_img = nib.Nifti1Image(first_data, first_img.affine, first_img.header)
    nib.save(modified_first_img, modified_label_map_path)

    print(f"Modified label map saved to {modified_label_map_path}")


def paste_ventricles_on_no_ventricles(no_ventricles_path, ventricles_mask_path, output_path, label=4):
    """
    Paste a ventricle mask on top of a no-ventricles label map with a specified label.

    Parameters:
    - no_ventricles_path: str, path to the label map without ventricles.
    - ventricles_mask_path: str, path to the ventricle mask to overlay.
    - output_path: str, path to save the resulting label map.
    - label: int, label to assign to the ventricles (default=4).
    """
    no_vent_img = nib.load(no_ventricles_path)
    no_vent_data = no_vent_img.get_fdata().copy()

    ventricles_img = nib.load(ventricles_mask_path)
    ventricles_data = ventricles_img.get_fdata()

    # Assign the label to the ventricle mask regions
    no_vent_data[ventricles_data > 0] = label

    # Save the new label map
    result_img = nib.Nifti1Image(no_vent_data, no_vent_img.affine, no_vent_img.header)
    nib.save(result_img, output_path)
    print(f"Pasted ventricles on no_ventricles map and saved to {output_path}")

import os





# File paths for 91-label map processing (existing functionality)
input_file_7_labels = f'{outputdir}/{sub}/orig_files/{sub}_7label_orig.nii.gz'  # Input NIfTI label map file
output_file_WM_left = f'{outputdir}/{sub}/hemis_splitted/{sub}_left_WM.nii.gz'   # Output file for first label set
output_file_WM_right = f'{outputdir}/{sub}/hemis_splitted/{sub}_right_WM.nii.gz'   # Output file for second label set
os.makedirs(os.path.dirname(output_file_WM_left), exist_ok=True)

path_seg_feta = f'{outputdir}/{sub}/orig_files/{sub}_7label_orig.nii.gz'
path_seg_wm_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_total_WM.nii.gz'
path_seg_ventricles_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_total_ventricles.nii.gz'

extract_wm_feta(path_seg_feta, path_seg_wm_feta)
extract_ventricles_feta(path_seg_feta, path_seg_ventricles_feta)

path_left_hemisphere_mask_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_left_WM.nii.gz'  # Output for modified 7-label map
path_right_hemisphere_mask_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_right_WM.nii.gz'

path_left_ventricles_mask_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_only_L_ventricles.nii.gz'  # Output for modified 7-label map
path_right_ventricles_mask_feta = f'{outputdir}/{sub}/hemis_splitted/{sub}_only_R_ventricles.nii.gz'  # Output for modified 7-label map

get_hemisphere_masks(path_seg_wm_feta, path_left_hemisphere_mask_feta, path_right_hemisphere_mask_feta)
get_hemisphere_masks(path_seg_ventricles_feta, path_left_ventricles_mask_feta, path_right_ventricles_mask_feta)

# Load the mask
mask_img = nib.load(path_left_hemisphere_mask_feta)
mask_data = mask_img.get_fdata()

# Set all non-zero values to 4
mask_data[mask_data > 0] = 4

# Save the new mask
new_mask_img = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
nib.save(new_mask_img, path_left_hemisphere_mask_feta)

# Load the mask
mask_img = nib.load(path_right_hemisphere_mask_feta)
mask_data = mask_img.get_fdata()

# Set all non-zero values to 4
mask_data[mask_data > 0] = 4

# Save the new mask
new_mask_img = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
nib.save(new_mask_img, path_right_hemisphere_mask_feta)

path_left_ventricles_mask_feta_derotated_orig_final = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_L_ventricles.nii.gz'
path_right_ventricles_mask_feta_derotated_orig_final = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_only_R_ventricles.nii.gz'
modified_first_img_no_ventricles_path = f'{outputdir}/{sub}/hemis_splitted/{sub}_7label_no_ventricles.nii.gz'  # Output for modified 7-label map

modify_label_map(path_seg_feta, path_left_ventricles_mask_feta, path_left_ventricles_mask_feta_derotated_orig_final,modified_first_img_no_ventricles_path)
modify_label_map(path_seg_feta, path_right_ventricles_mask_feta, path_right_ventricles_mask_feta_derotated_orig_final,modified_first_img_no_ventricles_path)
# breakpoint()
print(f'{sub}_splitted_hemis')
