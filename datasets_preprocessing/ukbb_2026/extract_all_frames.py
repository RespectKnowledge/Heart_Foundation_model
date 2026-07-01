"""Extract all cine frames from UKBB SAX data and save as separate files.

Current script saves only frame 0 (ED frame).
This script saves all 50 frames per subject.

Output naming convention:
    {subject}_sax_frame{frame_idx:02d}.nii.gz
    e.g. 1234567_sax_frame00.nii.gz ... 1234567_sax_frame49.nii.gz

This allows ConsisMAE SSL to see full cardiac motion cycle.
"""

import os
import nibabel as nib
from multiprocessing import Pool, cpu_count

input_root = "/mnt/all_data/Foundation_MRI_datasets/UKBB_2025"
output_dir = "/mnt/all_data/ssl_foundation/data/nnssl_raw/Dataset703_UKBB_MRI/imagesTr/"
os.makedirs(output_dir, exist_ok=True)

# get all subject directories
subjects = [s for s in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, s))]
print(f"Total subjects found: {len(subjects)}")


def process_subject_all_frames(subject):
    """Extract all cine frames from one subject's SAX volume."""
    patient_dir = os.path.join(input_root, subject)
    input_img   = os.path.join(patient_dir, "sax.nii.gz")

    if not os.path.exists(input_img):
        return f"skip {subject} — no sax.nii.gz"

    try:
        img = nib.load(input_img)

        # check shape — expected [X, Y, Z, T] where T = n_frames (usually 50)
        shape = img.shape
        if len(shape) < 4:
            # already a 3D volume — only one frame available
            output_path = os.path.join(output_dir, f"{subject}_sax_frame00.nii.gz")
            nib.save(img, output_path)
            return f"done {subject} — 3D only (1 frame)"

        n_frames = shape[3]
        saved    = 0

        # lazy load — avoid loading full 4D volume into memory at once
        data = img.dataobj

        for frame_idx in range(n_frames):
            output_path = os.path.join(output_dir, f"{subject}_sax_frame{frame_idx:02d}.nii.gz")

            # skip if already exists (allows resuming interrupted runs)
            if os.path.exists(output_path):
                saved += 1
                continue

            # extract single frame [X, Y, Z]
            frame_data = data[:, :, :, frame_idx]
            new_img    = nib.Nifti1Image(frame_data, img.affine, img.header)
            nib.save(new_img, output_path)
            saved += 1

        return f"done {subject} — {saved}/{n_frames} frames saved"

    except Exception as e:
        return f"error {subject}: {e}"


def process_subject_sampled_frames(subject, n_sample=8):
    """Extract sampled cine frames — memory efficient alternative.
    Saves n_sample evenly spaced frames instead of all 50.
    Recommended if storage is limited.
    """
    patient_dir = os.path.join(input_root, subject)
    input_img   = os.path.join(patient_dir, "sax.nii.gz")

    if not os.path.exists(input_img):
        return f"skip {subject} — no sax.nii.gz"

    try:
        img      = nib.load(input_img)
        shape    = img.shape
        data     = img.dataobj

        if len(shape) < 4:
            output_path = os.path.join(output_dir, f"{subject}_sax_frame00.nii.gz")
            nib.save(img, output_path)
            return f"done {subject} — 3D only"

        n_frames = shape[3]
        # evenly spaced frame indices e.g. [0, 6, 12, 18, 24, 30, 36, 43] for n_sample=8
        import numpy as np
        frame_indices = np.linspace(0, n_frames - 1, n_sample, dtype=int).tolist()

        for frame_idx in frame_indices:
            output_path = os.path.join(output_dir, f"{subject}_sax_frame{frame_idx:02d}.nii.gz")
            if os.path.exists(output_path):
                continue
            frame_data = data[:, :, :, frame_idx]
            new_img    = nib.Nifti1Image(frame_data, img.affine, img.header)
            nib.save(new_img, output_path)

        return f"done {subject} — {len(frame_indices)} frames saved {frame_indices}"

    except Exception as e:
        return f"error {subject}: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "sampled"], default="all",
                        help="all=save all 50 frames, sampled=save 8 evenly spaced frames")
    parser.add_argument("--n_sample", type=int, default=8,
                        help="number of frames to sample (only used with --mode sampled)")
    args = parser.parse_args()

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} CPUs")
    print(f"Mode: {args.mode}")
    print(f"Subjects: {len(subjects)}")

    if args.mode == "all":
        fn = process_subject_all_frames
        print(f"Saving ALL frames per subject")
        print(f"Estimated output: {len(subjects)} × 50 = {len(subjects)*50:,} files")
    else:
        import functools
        fn = functools.partial(process_subject_sampled_frames, n_sample=args.n_sample)
        print(f"Saving {args.n_sample} sampled frames per subject")
        print(f"Estimated output: {len(subjects)} × {args.n_sample} = {len(subjects)*args.n_sample:,} files")

    errors = []
    skipped = []
    done = []

    with Pool(workers) as p:
        for i, result in enumerate(p.imap_unordered(fn, subjects), 1):
            if i % 500 == 0:
                print(f"Progress: {i}/{len(subjects)} ({i/len(subjects)*100:.1f}%)")
            if result.startswith("error"):
                errors.append(result)
                print(result)
            elif result.startswith("skip"):
                skipped.append(result)
            else:
                done.append(result)

    print(f"\n{'='*50}")
    print(f"Done    : {len(done)}")
    print(f"Skipped : {len(skipped)}")
    print(f"Errors  : {len(errors)}")
    if errors:
        print("First 5 errors:")
        for e in errors[:5]:
            print(f"  {e}")

    # estimate storage used
    print(f"\nEstimated files created: {len(done) * (50 if args.mode == 'all' else args.n_sample):,}")
