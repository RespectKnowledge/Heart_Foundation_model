import nibabel as nib
import numpy as np
import os

images_folder = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset702_Public_MRI/imagesTr"
nii_files = [f for f in os.listdir(images_folder) if f.endswith(".nii.gz")]

fixed_count = 0

for f in nii_files:
    file_path = os.path.join(images_folder, f)
    try:
        # Try loading with nibabel
        img = nib.load(file_path)
    except Exception as e:
        print(f"Cannot load {f} with nibabel: {e}")
        continue

    # Check if the affine is orthonormal
    affine = img.affine
    R = affine[:3, :3]
    # Compute orthonormality: R^T R should be identity
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        print(f"Fixing {f} ...")
        # Replace rotation part with identity
        new_affine = np.eye(4)
        new_affine[:3, 3] = affine[:3, 3]  # preserve translation
        new_img = nib.Nifti1Image(img.get_fdata(), new_affine, header=img.header)
        nib.save(new_img, file_path)
        fixed_count += 1

print(f"Done! Fixed {fixed_count} NIfTI files with non-orthonormal directions.")


#import nibabel as nib
#import numpy as np
#import os
#
#images_folder = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset702_Public_MRI/imagesTr"
#nii_files = [f for f in os.listdir(images_folder) if f.endswith(".nii.gz")]
#
#fixed_count = 0
#
#for f in nii_files:
#    file_path = os.path.join(images_folder, f)
#    try:
#        # Load the NIfTI file
#        img = nib.load(file_path)
#    except Exception as e:
#        print(f"Cannot load {f}: {e}")
#        continue
#
#    affine = img.affine
#    R = affine[:3, :3]
#
#    # Check if rotation is orthonormal (identity check)
#    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
#        print(f"Fixing {f} ...")
#        # Replace rotation part with identity but keep translation
#        new_affine = np.eye(4)
#        new_affine[:3, 3] = affine[:3, 3]  # preserve translation
#        new_img = nib.Nifti1Image(img.get_fdata(), new_affine, header=img.header)
#        nib.save(new_img, file_path)
#        fixed_count += 1
#
#print(f"Done! Fixed {fixed_count} NIfTI files with non-orthonormal directions.")