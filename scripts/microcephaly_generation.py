import numpy as np
import nibabel as nib
import os
import argparse
import SimpleITK as sitk

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-sub', '--subject', type=str, default="", help="Subject identifier")
parser.add_argument('--shrink_factor', type=float, help="Shrink factor for the label map")
parser.add_argument('--outputdir', type=str, default='', help="Output directory")
parser.add_argument('--input_file', type=str, required=True, help="Input file name")
parser.add_argument('--output_file', type=str, required=True, help="Output file name")

args = parser.parse_args()

subject = args.subject
shrink_factor = args.shrink_factor
outputdir = args.outputdir
input_file = args.input_file
output_file = args.output_file

def shrink_label_map_nii(label_map_nii, output_file, shrink_factor):
    """Shrink the label map by applying a scaling transformation."""
    label_map = label_map_nii.get_fdata()
    
    # Compute the centroid of the label map (center of non-zero voxels)
    non_zero_voxels = np.array(np.nonzero(label_map))
    centroid = np.mean(non_zero_voxels, axis=1)
    
    # Create a SimpleITK image from the label map
    label_map_sitk = sitk.GetImageFromArray(label_map)
    
    # Get the original spacing, origin, and direction
    original_spacing = label_map_sitk.GetSpacing()
    original_origin = label_map_sitk.GetOrigin()
    original_direction = label_map_sitk.GetDirection()

    # Define the scaling transformation
    transform = sitk.AffineTransform(3)
    transform.Scale([shrink_factor, shrink_factor, shrink_factor])
    
    # Convert centroid to physical space coordinates
    centroid_physical = label_map_sitk.TransformContinuousIndexToPhysicalPoint(centroid[::-1])
    
    # Define the translation to center the image on the centroid
    transform.Translate((np.array(centroid_physical) * (1 - shrink_factor)).tolist())
    
    # Apply the transformation
    label_map_shrunk_sitk = sitk.Resample(
        label_map_sitk, 
        label_map_sitk.GetSize(),
        transform,
        sitk.sitkNearestNeighbor,
        original_origin,
        original_spacing,
        original_direction,
        0,
        label_map_sitk.GetPixelID()
    )
    
    # Convert back to NumPy array
    label_map_shrunk = sitk.GetArrayFromImage(label_map_shrunk_sitk)

    # Create and save the shrunk label map
    label_map_shrunk_nii = nib.Nifti1Image(label_map_shrunk, label_map_nii.affine, label_map_nii.header)
    nib.save(label_map_shrunk_nii, output_file)

    # Modify label map
    modified_shrink_labelmap = np.where(label_map_shrunk == 0, 1, label_map_shrunk)
    result_labelmap = np.where(label_map == 0, 0, modified_shrink_labelmap)

    # Create an image marking the centroid
    centroid_image = np.zeros_like(label_map)
    centroid_voxel = tuple(centroid.astype(int))
    centroid_image[centroid_voxel] = 1
    centroid_image_nii = nib.Nifti1Image(centroid_image, label_map_nii.affine, label_map_nii.header)

    return nib.Nifti1Image(result_labelmap, label_map_nii.affine, label_map_nii.header), centroid_image_nii

# File paths
input_path = os.path.join(outputdir, input_file)
output_path = os.path.join(outputdir, output_file)
output_path_2 = os.path.join(outputdir, f"{output_file}_not_capped.nii.gz")
centroid_image_path = os.path.join(outputdir, f"{output_file}_centroid.nii.gz")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load and process the label map
label_map_nii = nib.load(input_path)
shrunk_label_map_nii, centroid_image_nii = shrink_label_map_nii(label_map_nii, output_path_2, shrink_factor)

# Save the processed label map
nib.save(shrunk_label_map_nii, output_path)
# Save the centroid image
nib.save(centroid_image_nii, centroid_image_path)

print(f'Saved shrunk label map to: {output_path}')
print(f'Saved centroid image to: {centroid_image_path}')
