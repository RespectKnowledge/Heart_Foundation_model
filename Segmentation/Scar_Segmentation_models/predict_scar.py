
# -*- coding: utf-8 -*-
"""
Optimized prediction script for Scar Segmentation using nnU-Net UxLSTM models.
"""

import os
import torch
import argparse
import time
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/input', help='Path to input directory containing NIfTI files.')
    parser.add_argument('--output_dir', type=str, default='/output', help='Path to output directory for predictions.')
    args = parser.parse_args()

    start_time = time.time()

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {args.output_dir}')

    # Path to the trained model folder
    resources_dir = '/home/aqayyum/Scar_Segmentation_models/'
    model_folder = join(resources_dir, 'nnUNet_results/Dataset199_LGE_MI/nnUNetTrainerUxLSTMEnc__nnUNetPlans__3d_fullres/')

    # Instantiate the predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        #device=torch.device('cpu'),  # CHANGE TO CPU
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # Initialize model from trained folder
    print("Initializing model...")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )

    # Iterate over all NIfTI files directly in the input directory
    for file in os.listdir(args.input_dir):
        if not file.endswith('.nii.gz'):
            continue

        file_path = os.path.join(args.input_dir, file)
        output_file_path = os.path.join(args.output_dir, file)

        try:
            # Read image
            img, props = SimpleITKIO().read_images([file_path])
            print(f"Processing file: {file_path}, img.shape: {img.shape}")

            # Generate prediction
            pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
            pred_array = pred_array.astype(np.uint8)
            print(f"pred_array.shape: {pred_array.shape}, labels: {np.unique(pred_array)}")

            # Convert to SimpleITK image and set metadata
            image = sitk.GetImageFromArray(pred_array)
            image.SetDirection(props['sitk_stuff']['direction'])
            image.SetOrigin(props['sitk_stuff']['origin'])
            image.SetSpacing(props['sitk_stuff']['spacing'])

            # Save prediction
            sitk.WriteImage(image, output_file_path, useCompression=True)
            print(f"Prediction saved at: {output_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    main()
