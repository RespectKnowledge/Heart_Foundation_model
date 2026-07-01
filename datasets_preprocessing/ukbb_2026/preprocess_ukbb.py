#import os
#import nibabel as nib
#
## Input patient directory
#patient_dir = "/mnt/all_data/Foundation_MRI_datasets/UKBB_2025/6015656_2/"
#
## Input image
#input_img_path = os.path.join(patient_dir, "sax.nii.gz")
#
## Output directory
#output_dir = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset701_ukbb/imagesTr"
#os.makedirs(output_dir, exist_ok=True)
#
#patient_id = os.path.basename(patient_dir)
#output_img_path = os.path.join(output_dir, f"{patient_id}_sax.nii.gz")
#
## Load image
#img = nib.load(input_img_path)
#data = img.get_fdata()
#
#print("Original shape:", data.shape)
#
## Take first time frame
#first_frame = data[:, :, :, 0]
#
#print("New shape:", first_frame.shape)
#
## Create new nifti (keep affine/header)
#new_img = nib.Nifti1Image(first_frame, img.affine, img.header)
#
## Save
#nib.save(new_img, output_img_path)
#
#print("Saved to:", output_img_path)


############################
import os
import nibabel as nib
from multiprocessing import Pool, cpu_count

input_root = "/mnt/all_data/Foundation_MRI_datasets/UKBB_2025"
output_dir = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset701_ukbb/imagesTr"

os.makedirs(output_dir, exist_ok=True)

subjects = [s for s in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, s))]


def process_subject(subject):

    patient_dir = os.path.join(input_root, subject)
    input_img = os.path.join(patient_dir, "sax.nii.gz")

    if not os.path.exists(input_img):
        return f"skip {subject}"

    try:
        img = nib.load(input_img)

        # Lazy loading (no full memory load)
        data = img.dataobj

        # Extract first time frame from 50 frames
        first_frame = data[:, :, :, 0]

        output_path = os.path.join(output_dir, f"{subject}_sax.nii.gz")

        new_img = nib.Nifti1Image(first_frame, img.affine, img.header)
        nib.save(new_img, output_path)

        return f"done {subject}"

    except Exception as e:
        return f"error {subject}: {e}"


if __name__ == "__main__":

    workers = cpu_count() - 1
    print("Using", workers, "CPUs")

    with Pool(workers) as p:
        for i, result in enumerate(p.imap_unordered(process_subject, subjects), 1):

            if i % 200 == 0:
                print(f"Processed {i}/{len(subjects)}")

            print(result)