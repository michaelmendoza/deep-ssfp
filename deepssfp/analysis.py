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

def create_brain_dataset(save_path = None):
    if save_path and os.path.isfile(save_path):
        print(f'Saved dataset found. Loading from file: {save_path}')
        dataset = np.load(save_path, allow_pickle=True)[0]
        return dataset

    dataset = mssfp.generate_ssfp_dataset(
        phantom_type='brain', 
        npcs=4, 
        f=500, 
        alpha=np.deg2rad(60), 
        sigma=0.001, 
        data_indices=[(0, 2), (120,180)], 
        rotation=90,
        useRotate=True, 
        useDeform=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, [dataset])
    return dataset

def create_block_dataset(save_path = None):
    if save_path and os.path.isfile(save_path):
        print(f'Saved dataset found. Loading from file: {save_path}')
        dataset = np.load(save_path, allow_pickle=True)[0]
        return dataset

    slices = 500
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
        shape=128, 
        ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        tissues = tissue_parameters,
        npcs=4, 
        f=500, 
        alpha=np.deg2rad(40), 
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
    ds_path = os.path.join(model_dir, f"{model_name}_dataset.npy")

    return path, ds_path

def run_training(dataset, mode='BandRemoval:4', model_name="block_phantom", model_dir="D:/DeepSSFP/", train_model=True, verbose=True) -> dict:
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
    path, ds_path = generate_paths(mode, model_name, model_dir)

    if verbose:
        print(f"\n===== Training: {mode} =====")
        print(f"Model path: {path}")
        print(f"Dataset path: {ds_path}")

        # Train or load model
    if train_model:
        if verbose:
            print("\nTraining model...")
        model, history_dict, ds, predictions = deepssfp.train(
            mode=mode,
            model_name=model_name,
            model_dir=model_dir,
            custom_dataset=dataset,
            epochs=800,
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
        'history': history_dict 
    }

