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

def run_experiment(mode='BandRemoval:4', model_name="block_phantom", model_dir="D:/DeepSSFP/", train_model=True, custom_dataset=None, save_dataset=True):
    """Run a single experiment with enhanced metrics collection.
    
    Parameters
    ----------
    mode : str
        Training mode from dataset.modes
    model_name : str
        Base name for the model
    model_dir : str
        Directory to save/load model weights and results
    train_model : bool
        Whether to train a new model or load an existing one
    custom_dataset : dict, optional
        Dictionary containing custom dataset
    save_dataset : bool, optional
        Whether to save the dataset to disk
        
    Returns
    -------
    dict
        Dictionary containing experiment results
    """
    print(f"\n===== Running experiment: {mode} =====")
    
    # Create paths
    path = os.path.join(model_dir, f"{model_name}_{mode.lower().replace(':', '_')}")
    ds_path = os.path.join(model_dir, f"{model_name}_dataset.npy")
    print(f"Model path: {path}")
    print(f"Dataset path: {ds_path}")

    # Load or generate dataset
    if custom_dataset is not None:
        dataset = custom_dataset
        if save_dataset:
            os.makedirs(os.path.dirname(ds_path), exist_ok=True)
            np.save(ds_path, [dataset])    
    elif os.path.isfile(ds_path):
        print(f'Saved dataset found. Loading from file: {ds_path}')
        dataset = np.load(ds_path, allow_pickle=True)[0]
    else:
        print('Generating new mock phantomdataset...')
        slices = 200
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
            alpha=np.deg2rad(60), 
            sigma=0.001, 
            data_indices=[(0, 2), (120,180)], 
            useRotate=True, 
            useDeform=True
        )
        if save_dataset:
            os.makedirs(os.path.dirname(ds_path), exist_ok=True)
            np.save(ds_path, [dataset])

    # Load dataset for experiment
    data = dataset['M']
    print(f"Data shape: {data.shape}")
    
    # Create experiment dataset
    ds = deepssfp.dataset.Dataset(mode, input_data=data)
    print(f"Training data shape: {ds.x_train.shape}, {ds.y_train.shape}")
    print(f"Input scaler stats: mean={ds.inputScaler.mean:.4f}, std={ds.inputScaler.std:.4f}")
    print(f"Output scaler stats: mean={ds.outputScaler.mean:.4f}, std={ds.outputScaler.std:.4f}")

    # Visualize dataset
    deepssfp.visualize_images(
        input_images=ds.x_test,
        target_images=ds.y_test,
        num_samples=3,
        kspace= True if mode == deepssfp.DataMode.SuperFOV.value else False,
    )

    # Train or load model
    if train_model:
        print("\nTraining model...")
        model, history_dict, ds, predictions = deepssfp.train(
            mode=mode,
            model_name=model_name,
            model_dir=model_dir,
            custom_dataset=ds,
            epochs=500,
            use_early_stopping=False,
            patience=200,
            continue_training=True
        )
    else:
        print("\nLoading pre-trained model...")
        try:
            model, history_dict = deepssfp.load_model(
                mode=mode,
                model_name=model_name,
                model_dir=model_dir
            )
        except FileNotFoundError:
            print("No pre-trained model found. Exiting ...")
            return
        
    # Plot training history if available
    if history_dict:
        deepssfp.plot_training_history(
            history_dict,
            title=f"{mode} Training Results",
            save_path=f"{path}_training_history.png"
        )

    # Reload dataset for consistent scaling across experiments
    #ds = deepssfp.Dataset(mode, input_data=data)
    seg = dataset['seg']    
    if len(seg.shape) == 3:
        seg = seg[ds.shuffled_indices]
        seg = seg[-ds.x_test.shape[0]:, :]

    # Generate predictions
    x_test = ds.inputScaler.inverse_transform(ds.x_test)
    y_test = ds.outputScaler.inverse_transform(ds.y_test)
    predictions = ds.outputScaler.inverse_transform(deepssfp.predict(model, ds.x_test))

    # Visualize results
    deepssfp.visualize_images(
        input_images=x_test,
        target_images=y_test,
        predicted_images=predictions,
        num_samples=2,
        kspace= True if mode == deepssfp.DataMode.SuperFOV.value else False,
        save_path=f"{path}_prediction_results.png"
    )

    # Form data for banding reduction evaluation
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
    print(f"Target shape: {target_complex.shape}, Prediction shape: {pred_complex.shape}")

    # Plot a line profile
    segment_ids = np.unique(dataset['seg'])

    plt.figure(figsize=(10, 5))
    line_count = math.floor(math.sqrt(len(segment_ids)))
    line_dx = 40 #64
    row_idx = 0
    line_idx = 20
    target_line = np.array([])
    pred_line = np.array([])
    for i in range(line_count):
        target_line = np.append(target_line, target_complex[row_idx, line_idx, :])
        pred_line = np.append(pred_line, pred_complex[row_idx, line_idx, :])
        line_idx += line_dx

    plt.plot(np.abs(target_line), label='Target')
    plt.plot(np.abs(pred_line), label='Prediction')
    plt.title(f"{mode} - Line Profile")
    plt.xlabel("Pixel Position")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}_line_profile.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Run band reduction evaluation
    print("\nEvaluating band reduction...")
    band_metrics = deepssfp.evaluate_band_reduction(
        target_complex, 
        pred_complex, 
        seg, 
        sort_values=False, 
        fig_size=(6, 2), 
        save_path=f"{path}_band_metrics"
    )

    # Calculate additional image quality metrics
    print("\nCalculating image quality metrics...")
    # Get unique segment IDs from segmentation mask
    segment_ids = np.unique(seg)
    # Calculate metrics for whole image and for each segment
    image_metrics = deepssfp.metrics.calculate_image_metrics(
        target_complex, 
        pred_complex,
        segment_ids=segment_ids,
        segmentation=seg
    )
    
    # Print metrics summary
    deepssfp.metrics.print_metrics_summary(image_metrics)
    
    # Save experiment results
    save_experiment_results(
        mode=mode,
        image_metrics=image_metrics,
        band_metrics=band_metrics,
        model_name=model_name,
        model_dir=model_dir
    )
    
    # Return results dictionary
    return {
        'mode': mode,
        'model': model,
        'dataset': ds,
        'predictions': pred_complex,
        'target': target_complex,
        'image_metrics': image_metrics,
        'band_metrics': band_metrics
    }

def run_all_experiments(modes=None, model_name="block_phantom", model_dir="D:/DeepSSFP/", train_models=True, custom_dataset=None):
    """Run all experiments and compare results.
    
    Parameters
    ----------
    modes : list, optional
        List of modes to run experiments on. If None, use default modes.
    model_name : str, optional
        Base name for the model
    model_dir : str, optional
        Directory to save/load models and results
    train_models : bool, optional
        Whether to train new models or load existing ones
    """
    if modes is None:
        modes = ['BandRemoval:2', 'BandRemoval:4', 'SuperFOV']
    
    # Run each experiment
    results = {}
    for mode in modes:
        results[mode] = run_experiment(
            mode=mode,
            model_name=model_name,
            model_dir=model_dir,
            train_model=train_models,
            custom_dataset=custom_dataset
        )
    
    '''
    # Compare experiment results
    compare_experiments(
        modes=modes,
        model_name=model_name,
        model_dir=model_dir
    )
    '''
    return results

def save_experiment_results(
    mode: str,
    image_metrics: Dict,
    band_metrics: Dict,
    model_name: str,
    model_dir: str
):
    """Save experiment results to files.
    
    Parameters
    ----------
    mode : str
        Experiment mode
    image_metrics : Dict
        Dictionary of image quality metrics
    band_metrics : Dict
        Dictionary of band reduction metrics
    model_name : str
        Base name for the model
    model_dir : str
        Directory to save results
    """
    # Create path
    path = os.path.join(model_dir, f"{model_name}_{mode.lower().replace(':', '_')}")
    
    # Save metrics to file
    all_metrics = {
        'mode': mode,
        'image_metrics': image_metrics,
        'band_metrics': band_metrics
    }
    np.save(f"{path}_all_metrics.npy", all_metrics)
    
    # Log success
    print(f"Metrics saved to {path}_all_metrics.npy")

def load_experiment_results(
    mode: str,
    model_name: str,
    model_dir: str
) -> Dict:
    """Load experiment results from file.
    
    Parameters
    ----------
    mode : str
        Experiment mode
    model_name : str
        Base name for the model
    model_dir : str
        Directory where results are saved
        
    Returns
    -------
    Dict
        Dictionary of metrics
    """
    # Create path
    path = os.path.join(model_dir, f"{model_name}_{mode.lower().replace(':', '_')}")
    
    # Check if file exists
    file_path = f"{path}_all_metrics.npy"
    if not os.path.exists(file_path):
        print(f"No metrics file found at {file_path}")
        return None
    
    # Load metrics
    all_metrics = np.load(file_path, allow_pickle=True).item()
    return all_metrics

def compare_experiments(
    modes: List[str],
    model_name: str,
    model_dir: str,
    metrics_to_compare: Optional[List[str]] = None,
    segment_ids: Optional[List[int]] = None
):
    """Compare results from multiple experiments.
    
    Parameters
    ----------
    modes : List[str]
        List of experiment modes to compare
    model_name : str
        Base name for the model
    model_dir : str
        Directory where results are saved
    metrics_to_compare : List[str], optional
        List of metrics to include in comparison
        If None, compare all available metrics
    segment_ids : List[int], optional
        List of segment IDs to include in comparison
        If None, only compare global metrics
    """
    # Set default metrics if not provided
    if metrics_to_compare is None:
        metrics_to_compare = ['mse', 'mae', 'psnr', 'ssim', 'nrmse']
    
    # Initialize data for comparison
    image_metrics_data = []
    band_metrics_data = []
    
    # Load results for each mode
    results = {}
    for mode in modes:
        result = load_experiment_results(mode, model_name, model_dir)
        if result is not None:
            results[mode] = result
    
    if not results:
        print("No results found for comparison.")
        return
    
    # Process image quality metrics
    for mode, result in results.items():
        image_metrics = result.get('image_metrics', {})
        
        # Global metrics
        if 'global' in image_metrics:
            for metric, value in image_metrics['global'].items():
                if metric in metrics_to_compare:
                    image_metrics_data.append({
                        'Mode': mode,
                        'Region': 'Global',
                        'Metric': metric,
                        'Value': value
                    })
        
        # Segment-specific metrics
        if segment_ids is not None:
            for id in segment_ids:
                segment_key = f'segment_{id}'
                if segment_key in image_metrics:
                    for metric, value in image_metrics[segment_key].items():
                        if metric in metrics_to_compare:
                            image_metrics_data.append({
                                'Mode': mode,
                                'Region': f'Segment {id}',
                                'Metric': metric,
                                'Value': value
                            })
    
    # Process band reduction metrics
    for mode, result in results.items():
        band_metrics = result.get('band_metrics', {})
        
        if band_metrics:
            for id, id_metrics in band_metrics.items():
                for metric, value in id_metrics.items():
                    # Only include numeric metrics
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        band_metrics_data.append({
                            'Mode': mode,
                            'ID': id,
                            'Metric': metric,
                            'Value': value
                        })
    
    # Convert to dataframes
    image_df = pd.DataFrame(image_metrics_data)
    band_df = pd.DataFrame(band_metrics_data) if band_metrics_data else None
    
    # Save to CSV
    image_df.to_csv(f"{model_dir}/{model_name}_image_quality_comparison.csv", index=False)
    if band_df is not None:
        band_df.to_csv(f"{model_dir}/{model_name}_band_reduction_comparison.csv", index=False)
    
    # Create comparison plots
    create_comparison_plots(image_df, band_df, model_name, model_dir)
    
    # Print summary
    print_comparison_summary(image_df, band_df, model_name, model_dir)

def create_comparison_plots(
    image_df: pd.DataFrame,
    band_df: Optional[pd.DataFrame],
    model_name: str,
    model_dir: str
):
    """Create plots to compare metrics across experiment modes.
    
    Parameters
    ----------
    image_df : pd.DataFrame
        DataFrame of image quality metrics
    band_df : pd.DataFrame, optional
        DataFrame of band reduction metrics
    model_name : str
        Base name for the model
    model_dir : str
        Directory to save plots
    """
    # Set plot style
    sns.set(style="whitegrid")
    
    # Create folder for plots if it doesn't exist
    plots_dir = os.path.join(model_dir, f"{model_name}_comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # ==== Image Quality Metrics Plots ====
    
    # Plot all metrics by mode
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        x='Mode', 
        y='Value', 
        hue='Region', 
        col='Metric', 
        data=image_df, 
        kind='bar',
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharex=True,
        sharey=False
    )
    g.set_axis_labels("Mode", "Value")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "image_metrics_by_mode.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot each metric separately
    for metric in image_df['Metric'].unique():
        plt.figure(figsize=(10, 6))
        metric_df = image_df[image_df['Metric'] == metric]
        sns.barplot(x='Mode', y='Value', hue='Region', data=metric_df)
        plt.title(f"{metric.upper()} Comparison")
        plt.ylabel(metric.upper())
        plt.xlabel("Mode")
        plt.legend(title="Region")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{metric}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # ==== Band Reduction Metrics Plots ====
    if band_df is not None:
        # Create plots for relevant band reduction metrics
        important_metrics = ['pred_ripple', 'pred_snr', 'mse', 'mae']
        for metric in important_metrics:
            if metric in band_df['Metric'].values:
                plt.figure(figsize=(12, 6))
                subset = band_df[band_df['Metric'] == metric]
                sns.barplot(x='ID', y='Value', hue='Mode', data=subset)
                plt.title(f"{metric.upper()} Comparison by Segment ID")
                plt.xlabel("Segment ID")
                plt.ylabel(metric.upper())
                plt.legend(title="Mode")
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"band_{metric}_by_id.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    # ==== Combined Dashboard ====
    create_metrics_dashboard(image_df, model_name, plots_dir)
    
    print(f"Comparison plots saved to {plots_dir}")

def create_metrics_dashboard(image_df, model_name, plots_dir):
    """Create a dashboard of all image quality metrics.
    
    Parameters
    ----------
    image_df : pd.DataFrame
        DataFrame of image quality metrics
    model_name : str
        Base name for the model
    plots_dir : str
        Directory to save plots
    """
    # Filter for global metrics only to simplify dashboard
    global_df = image_df[image_df['Region'] == 'Global']
    
    # Get unique metrics and modes
    metrics = global_df['Metric'].unique()
    modes = global_df['Mode'].unique()
    
    # Create bar chart to compare all metrics across modes
    num_metrics = len(metrics)
    
    # Set up the figure with a grid of subplots
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 2))
    
    # Ensure axes is always a list even with a single metric
    if num_metrics == 1:
        axes = [axes]
    
    # Create a subplot for each metric
    for i, metric in enumerate(metrics):
        metric_df = global_df[global_df['Metric'] == metric].copy()
        
        # Handle metric direction (lower or higher is better)
        if metric.lower() in ['mse', 'mae', 'nrmse']:
            # Add a note to the label
            metric_label = f"{metric.upper()} (lower is better)"
        else:
            metric_label = f"{metric.upper()} (higher is better)"
        
        # Create bar plot only for non-NaN values
        valid_data = metric_df.dropna(subset=['Value'])
        if not valid_data.empty:
            sns.barplot(x='Mode', y='Value', data=valid_data, ax=axes[i])
            
            # Add value labels on top of bars
            for j, mode in enumerate(valid_data['Mode'].unique()):
                val = valid_data[valid_data['Mode'] == mode]['Value'].values[0]
                axes[i].text(j, val, f"{val:.4f}", ha='center', va='bottom', fontsize=9)
        
        axes[i].set_title(metric_label)
        axes[i].set_xlabel('')
        
        # Set consistent y-axis limits for same metric types
        if metric.lower() in ['mse', 'mae', 'nrmse']:
            non_nan_values = metric_df['Value'].dropna().values
            if len(non_nan_values) > 0:
                axes[i].set_ylim(0, max(non_nan_values) * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def print_comparison_summary(
    image_df: pd.DataFrame,
    band_df: Optional[pd.DataFrame],
    model_name: str,
    model_dir: str
):
    """Print a summary of the comparison.
    
    Parameters
    ----------
    image_df : pd.DataFrame
        DataFrame of image quality metrics
    band_df : pd.DataFrame, optional
        DataFrame of band reduction metrics
    model_name : str
        Base name for the model
    model_dir : str
        Directory where results are saved
    """
    # Filter for global metrics and get unique modes and metrics
    global_df = image_df[image_df['Region'] == 'Global']
    modes = global_df['Mode'].unique()
    metrics = global_df['Metric'].unique()
    
    # Create summary table
    summary_table = []
    for metric in metrics:
        row = {'Metric': metric.upper()}
        for mode in modes:
            value = global_df[(global_df['Mode'] == mode) & (global_df['Metric'] == metric)]['Value'].values
            if len(value) > 0:
                row[mode] = value[0]
            else:
                row[mode] = np.nan
        summary_table.append(row)
    
    # Convert to DataFrame and print
    summary_df = pd.DataFrame(summary_table)
    
    print("\n===== Experiment Comparison Summary =====")
    print(f"Model: {model_name}")
    print(f"Modes: {', '.join(modes)}")
    print("\nGlobal Image Quality Metrics:")
    print(summary_df.to_string(index=False))
    
    # Identify best mode for each metric
    print("\nBest Mode by Metric:")
    for metric in metrics:
        metric_df = summary_df[summary_df['Metric'] == metric.upper()].iloc[0].drop('Metric')
        
        # Convert to numeric and drop NaN values
        valid_values = pd.to_numeric(metric_df, errors='coerce').dropna()
        
        if valid_values.empty:
            print(f"  {metric.upper()}: No valid data available")
            continue
            
        # For metrics where lower is better
        if metric.lower() in ['mse', 'mae', 'nrmse']:
            best_mode = valid_values.idxmin()
            best_value = valid_values.min()
            print(f"  {metric.upper()}: {best_mode} ({best_value:.6f})")
        # For metrics where higher is better
        else:
            best_mode = valid_values.idxmax()
            best_value = valid_values.max()
            print(f"  {metric.upper()}: {best_mode} ({best_value:.6f})")
    
    # Save summary to file
    summary_df.to_csv(f"{model_dir}/{model_name}_metrics_summary.csv", index=False)
    print(f"\nSummary saved to {model_dir}/{model_name}_metrics_summary.csv")

if __name__ == "__main__":
    # Configure output directory
    model_dir = "D:/DeepSSFP/"
    model_name="brain_phantom"
    
    # Run individual experiments
    # run_experiment(mode='BandRemoval:2', model_dir=model_dir)
    # run_experiment(mode='BandRemoval:4', model_dir=model_dir)
    # run_experiment(mode='SuperFOV', model_dir=model_dir)
    
    # Or run all experiments and compare
    '''
    run_all_experiments(
        modes=['BandRemoval:2', 'BandRemoval:4', 'SuperFOV'],
        model_name="block_phantom",
        model_dir=model_dir,
        train_models=True  # Set to False to use pre-trained models
    )'
    '''
    run_experiment(
        mode=deepssfp.DataMode.SyntheticBanding.value, 
        model_name=model_name,
        model_dir=model_dir,
        train_model=True,
        #custom_dataset=dataset
    )
