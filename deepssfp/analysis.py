"""
Module for comparing DeepSSFP experiment results.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import mssfp
import deepssfp
import deepssfp.recon
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

def create_brain_rawdata(save_path = None):
    if save_path and os.path.isfile(save_path):
        print(f'Saved dataset found. Loading from file: {save_path}')
        dataset = np.load(save_path, allow_pickle=True)[0]
        return dataset

    dataset = mssfp.generate_ssfp_dataset(
        phantom_type='brain', 
        npcs=4, 
        TR = 3e-3, 
        TE = 3e-3 / 2, 
        f=500, 
        df = 0.75 * 1/3e-3,
        df_window = 1,
        fn_perlin=100, 
        fn_perlin_size=3,
        alpha=np.deg2rad(22), 
        sigma=0.001, 
        data_indices=[(0, 8), (120,180)], 
        rotation=90,
        useRotate=True, 
        useDeform=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, [dataset])
    return dataset

def create_block_rawdata(save_path = None):
    if save_path and os.path.isfile(save_path):
        print(f'Saved dataset found. Loading from file: {save_path}')
        dataset = np.load(save_path, allow_pickle=True)[0]
        return dataset

    slices = 480
    tissue_parameters = {
        0: ('none', 0, 0, 0),
        1: ('fat', 0.350, 0.130, 0),
        2: ('bone marrow', 0.370, 0.05, 0),
        3: ('liver', 0.8, 0.04, 0),
        4: ('white matter', 1.0, 0.08, 0),
        5: ('myocardium', 1.150, 0.045, 0),
        6: ('vessels', 1.2, 0.05, 0),   
        7: ('gray matter', 1.3, 0.110, 0),
        8: ('muscle', 1.4, 0.030, 0),
        9: ('CSF', 4.0, 1.0, 0)
    }

    dataset = mssfp.generate_ssfp_dataset(
        phantom_type='block', 
        slices=slices, 
        shape=256, 
        ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        tissues = tissue_parameters,
        npcs=4, 
        f=500, 
        df_window = 1,
        fn_perlin=100, 
        fn_perlin_size=3,
        alpha=np.deg2rad(22), 
        sigma=0.001, 
        useRotate=True, 
        useDeform=True,
        rotation=90,
        fn_offset=250
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, [dataset])
    return dataset
    
def generate_paths(mode, model_name, model_dir):
    """Generate paths for a model and dataset."""
    # Create paths
    path = os.path.join(model_dir, f"{model_name}_{mode.lower().replace(':', '_')}")
    return path

def run_training(dataset, mode='BandRemoval:4', model_name="block_phantom", model_dir="D:/DeepSSFP/", 
                 fine_tune=False, fine_tune_model_path="", fine_tune_suffix="finetuned", fine_tune_lr=1e-5,
                 train_model=True, epochs=400, verbose=True) -> dict:
    """ Train a deepssfp model on a dataset.
    
    Parameters
    ----------
    dataset : deepssfp.dataset.Dataset
        Dataset used for training of type (deepssfp.dataset.Dataset)
    mode : str
        Training mode from dataset.modes
    model_name : str
        Base name for the model
    model_dir : str
        Directory to save/load model weights and results
    train_model : bool
        Whether to train a new model or load an existing one
        
    Returns
    -------
    dict
        Dictionary containing experiment results
    """
    
    # Create paths
    path = generate_paths(mode, model_name, model_dir)

    if verbose:
        print(f"\n===== Training: {mode} =====")
        print(f"Model path: {path}")

    # Train or load model
    if train_model:
        if verbose:
            print("\nTraining model...")

        if fine_tune:
            if verbose:
                print("\nFine-tuning model...")
            model, history_dict, ds, predictions = deepssfp.train(
                mode=mode,
                model_name=model_name,
                model_dir=model_dir,
                custom_dataset=dataset,
                epochs=epochs,
                use_early_stopping=True,
                patience=200,
                continue_training=True,
                fine_tune=True,
                fine_tune_model_path=fine_tune_model_path,
                fine_tune_suffix=fine_tune_suffix,
                fine_tune_lr=fine_tune_lr
            )
        else:
            model, history_dict, ds, predictions = deepssfp.train(
                mode=mode,
                model_name=model_name,
                model_dir=model_dir,
                custom_dataset=dataset,
                epochs=epochs,
                use_early_stopping=True,
                patience=200,
                continue_training=True
            )
    else:
        if verbose:
            print("\nLoading pre-trained model...")
        try:
            model, history_dict = deepssfp.load_model(
                mode=mode,
                model_name=model_name,
                model_dir=model_dir
            )
        except FileNotFoundError:
            print("No pre-trained model found. Exiting ...")
            return { 'error': "No pre-trained model found." }
        
    # Plot training history if available
    if verbose and history_dict:
        deepssfp.plot_training_history(
            history_dict,
            title=f"{mode} Training Results",
            save_path=f"{path}_training_history.png"
        )

    # Return results dictionary
    return {
        'mode': mode,
        'model': model,
        'history': history_dict, 
        'path': path
    }

def run_inference_on_test_dataset(dataset, model, verbose=False):
    mode = dataset.mode

    # Generate predictions
    x_test = dataset.inputScaler.inverse_transform(dataset.x_test)
    y_test = dataset.outputScaler.inverse_transform(dataset.y_test)
    predictions = dataset.outputScaler.inverse_transform(deepssfp.predict(model, dataset.x_test))

    if verbose:
        deepssfp.visualize_images(
            input_images=x_test,
            target_images=y_test,
            predicted_images=predictions,
            num_samples=2,
            kspace= True if mode == deepssfp.DataMode.SuperFOV.value else False
        )

    # Format data from real/imag pairs to complex
    if (mode == deepssfp.DataMode.SyntheticBanding.value):
        target = deepssfp.combine_synthetic_banding_datasets(x_test, y_test)
        target_complex = deepssfp.recon.gs_recon_3d(target)
        combine = deepssfp.combine_synthetic_banding_datasets(x_test, predictions)
        pred_complex = deepssfp.recon.gs_recon_3d(combine)
    else:
        pred_complex = deepssfp.from_pairs_to_complex(predictions)
        target_complex = deepssfp.from_pairs_to_complex(y_test)
    if (mode == deepssfp.DataMode.SuperFOV.value):
         pred_complex = deepssfp.ifft(pred_complex)
         target_complex = deepssfp.ifft(target_complex)

    return pred_complex, target_complex

def create_segmentation_mask(M):
    # Create mask of phantom
    _ = np.sqrt(np.sum(np.abs(M)**2, axis=3))
    _ = abs(_)
    from skimage.filters import threshold_li
    thresh = threshold_li(_)
    mask = np.abs(_) > thresh
    seg = mask * 1
    return seg

def compute_image_metrics(pred, target, verbose=False):

    metrics = {
        'mse': { 'values': [], 'mean': 0, 'std': 0 },
        'mae': { 'values': [], 'mean': 0, 'std': 0 },
        'psnr': { 'values': [], 'mean': 0, 'std': 0 },
        'ssim': { 'values': [], 'mean': 0, 'std': 0 },
        'nrmse': { 'values': [], 'mean': 0, 'std': 0 },
    }

    for i in range(target.shape[0]):
        _pred = np.abs(pred[i])
        _target = np.abs(target[i])
        data_range = np.max(_target) - np.min(_target)

        mse_val = np.mean((_target - _pred) ** 2)
        mae_val = np.mean(np.abs(_target - _pred))
        psnr_val = psnr(_target, _pred, data_range=data_range)
        ssim_val = ssim(_target, _pred, data_range=data_range, gaussian_weights=True)
        nrmse_val = nrmse(_target, _pred)

        # Add to metrics
        metrics['mse']['values'].append(mse_val)
        metrics['mae']['values'].append(mae_val)
        metrics['psnr']['values'].append(psnr_val)
        metrics['ssim']['values'].append(ssim_val)
        metrics['nrmse']['values'].append(nrmse_val)

    # Calculate mean and std
    status = ''
    for metric in metrics:
        metrics[metric]['mean'] = np.mean(metrics[metric]['values'])
        metrics[metric]['std'] = np.std(metrics[metric]['values'])
        #status += f"{metric}: {metrics[metric]['mean']:.2e} +/- {metrics[metric]['std']:.2e} "
    
        mean_str = f"{metrics[metric]['mean']:.4f}" if 1 <= abs(metrics[metric]['mean']) < 10000 else f"{metrics[metric]['mean']:.2e}"
        std_str = f"{metrics[metric]['std']:.4f}" if 1 <= abs(metrics[metric]['std']) < 10000 else f"{metrics[metric]['std']:.2e}"   
        status += f"{metric}: {mean_str} +/- {std_str} "

    if verbose:
        print(status)

    return metrics