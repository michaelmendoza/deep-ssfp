import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def evaluate_band_reduction(target, prediction, seg, sort_values=True, fig_size=(8, 3), save_path:Optional[str] = None):
    """
    Evaluate band reduction using segmentation masks and plot comparison 
    of target vs prediction for each segmentation ID.
    
    Parameters
    ----------
    target : np.ndarray
        The ground truth target data (complex array) with shape (samples, height, width)
    prediction : np.ndarray
        The model prediction data (complex array) with shape (samples, height, width)
    seg : np.ndarray
        Segmentation mask with integer labels with shape (height, width)
    sort_values : bool, optional
        Whether to sort values in scatter plots (default: True)
    fig_size : tuple, optional
        Size of each subplot (width, height) in inches
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    dict
        Dictionary of metrics by segmentation ID
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Print shapes for debugging
    print(f"Target shape: {target.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Segmentation shape: {seg.shape}")
    
    # Make sure target and prediction are the same shape
    if target.shape != prediction.shape:
        raise ValueError(f"Target shape {target.shape} and prediction shape {prediction.shape} must match")
    
    # Ensure we have a batch/samples dimension
    if len(target.shape) == 3 and len(seg.shape) == 2:
        if target.shape[1:] != seg.shape:
            raise ValueError(f"Target dimensions {target.shape[1:]} and segmentation dimensions {seg.shape} must match")
    else:
        raise ValueError("Expected target/prediction with shape (samples, height, width) and seg with shape (height, width)")
    
    # Process the first sample for visualization
    sample_idx = 0
    target_sample = target[sample_idx]
    prediction_sample = prediction[sample_idx]
    batch_size = target.shape[0]
    
    # Get unique segmentation IDs
    ids = np.unique(seg)
    print(f"Unique segmentation IDs: {ids}")
    
    # Filter out ID=0 (noise) for plotting, but keep it for SNR calculation
    plot_ids = np.array([id for id in ids if id != 0])
    if len(plot_ids) == 0:
        plot_ids = ids  # If no non-zero IDs, use all IDs
    
    print(f"IDs to plot: {plot_ids}")
    
    # Limit the number of plots to avoid creating too many subplots
    max_plots = 20  # Arbitrary limit to avoid too many plots
    if len(plot_ids) > max_plots:
        print(f"Warning: Too many IDs to plot ({len(plot_ids)}). Limiting to first {max_plots}.")
        plot_ids = plot_ids[:max_plots]
    
    # Create a figure with subplots - one row per ID to plot
    n_plot_ids = len(plot_ids)
    if n_plot_ids == 0:
        print("No valid segmentation IDs found. Please check your segmentation mask.")
        return {}
    
    ncols = math.ceil(math.sqrt(n_plot_ids))
    nrows = math.ceil(n_plot_ids / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_size[0] * ncols, fig_size[1] * nrows), squeeze=False)
    
    # Initialize metrics dictionaries
    metrics = {}
    all_sample_metrics = {}
    for id in ids: 
        all_sample_metrics[id] = {
            # Error metrics
            'mse': [],
            'mae': [],
            
            # Prediction metrics
            'pred_min': [],
            'pred_max': [],
            'pred_ripple': [],
            'pred_signal_power': [],
            
            # Target metrics
            'target_min': [],
            'target_max': [],
            'target_ripple': [],
            'target_signal_power': []
        }
    
    # Process all samples for metrics calculation
    for i in range(batch_size):
        target_i = target[i]
        prediction_i = prediction[i]
        
        # First, extract noise values if ID=0 exists
        pred_noise_power = None
        target_noise_power = None
        if 0 in np.unique(seg):
            noise_mask = (seg == 0)
            if np.sum(noise_mask) > 0:
                # For prediction noise
                pred_noise_values = np.abs(prediction_i[noise_mask])
                pred_noise_power = np.mean(pred_noise_values**2)
                
                # For target noise
                target_noise_values = np.abs(target_i[noise_mask])
                target_noise_power = np.mean(target_noise_values**2)
        
        for id in ids:
            mask = (seg == id)
            if np.sum(mask) == 0:
                continue
                
            target_values = np.abs(target_i[mask])
            pred_values = np.abs(prediction_i[mask])
            
            # Calculate error metrics
            mse = np.mean((target_values - pred_values) ** 2)
            mae = np.mean(np.abs(target_values - pred_values))
            
            # Calculate prediction metrics
            pred_min_val = np.min(pred_values)
            pred_max_val = np.max(pred_values)
            pred_mean_val = np.mean(pred_values)
            pred_ripple = 100 * (pred_max_val - pred_min_val) / pred_mean_val if pred_mean_val > 0 else 0
            pred_signal_power = np.mean(pred_values**2)
            
            # Calculate target metrics
            target_min_val = np.min(target_values)
            target_max_val = np.max(target_values)
            target_mean_val = np.mean(target_values)
            target_ripple = 100 * (target_max_val - target_min_val) / target_mean_val if target_mean_val > 0 else 0
            target_signal_power = np.mean(target_values**2)
            
            # Add to our all_sample_metrics collection
            all_sample_metrics[id]['mse'].append(mse)
            all_sample_metrics[id]['mae'].append(mae)
            
            all_sample_metrics[id]['pred_min'].append(pred_min_val)
            all_sample_metrics[id]['pred_max'].append(pred_max_val)
            all_sample_metrics[id]['pred_ripple'].append(pred_ripple)
            all_sample_metrics[id]['pred_signal_power'].append(pred_signal_power)
            
            all_sample_metrics[id]['target_min'].append(target_min_val)
            all_sample_metrics[id]['target_max'].append(target_max_val)
            all_sample_metrics[id]['target_ripple'].append(target_ripple)
            all_sample_metrics[id]['target_signal_power'].append(target_signal_power)
    
    # Create plots and calculate average metrics
    for i, id in enumerate(plot_ids):
        # Skip if no valid metrics
        if id not in all_sample_metrics or not all_sample_metrics[id]['mse']:
            continue
            
        # Use first sample for visualization
        mask = (seg == id)
        target_values = np.abs(target_sample[mask])
        pred_values = np.abs(prediction_sample[mask])
        
        if sort_values:
            # Sort values for better visualization
            sorted_indices = np.argsort(target_values)
            target_values = target_values[sorted_indices]
            pred_values = pred_values[sorted_indices]
        
        # Calculate average metrics across all samples
        avg_mse = np.mean(all_sample_metrics[id]['mse'])
        avg_mae = np.mean(all_sample_metrics[id]['mae'])
        
        # Prediction metrics
        avg_pred_min = np.mean(all_sample_metrics[id]['pred_min'])
        avg_pred_max = np.mean(all_sample_metrics[id]['pred_max'])
        avg_pred_ripple = np.mean(all_sample_metrics[id]['pred_ripple'])
        avg_pred_signal_power = np.mean(all_sample_metrics[id]['pred_signal_power'])
        
        # Target metrics
        avg_target_min = np.mean(all_sample_metrics[id]['target_min'])
        avg_target_max = np.mean(all_sample_metrics[id]['target_max'])
        avg_target_ripple = np.mean(all_sample_metrics[id]['target_ripple'])
        avg_target_signal_power = np.mean(all_sample_metrics[id]['target_signal_power'])
        
        # Calculate SNR if we have noise power (ID=0)
        pred_snr = None
        target_snr = None
        if 0 in np.unique(seg) and id != 0:
            noise_mask = (seg == 0)
            if np.sum(noise_mask) > 0:
                # Calculate average prediction noise power across all samples
                pred_noise_powers = []
                for j in range(batch_size):
                    noise_values = np.abs(prediction[j][noise_mask])
                    pred_noise_powers.append(np.mean(noise_values**2))
                avg_pred_noise_power = np.mean(pred_noise_powers)
                
                # Calculate average target noise power across all samples
                target_noise_powers = []
                for j in range(batch_size):
                    noise_values = np.abs(target[j][noise_mask])
                    target_noise_powers.append(np.mean(noise_values**2))
                avg_target_noise_power = np.mean(target_noise_powers)
                
                # Calculate SNRs
                if avg_pred_noise_power > 0:
                    pred_snr = 10 * np.log10(avg_pred_signal_power / avg_pred_noise_power)
                
                if avg_target_noise_power > 0:
                    target_snr = 10 * np.log10(avg_target_signal_power / avg_target_noise_power)
        
        # Store metrics
        metrics[id] = {
            'mse': avg_mse,
            'mae': avg_mae,
            
            'pred_min': avg_pred_min,
            'pred_max': avg_pred_max,
            'pred_ripple': avg_pred_ripple,
            'pred_snr': pred_snr,
            
            'target_min': avg_target_min,
            'target_max': avg_target_max,
            'target_ripple': avg_target_ripple,
            'target_snr': target_snr,
            
            'samples': len(all_sample_metrics[id]['mse']),
            'pixels': len(target_values)
        }
        
        # Create scatter plot
        ax = axs[i // ncols, i % ncols]

        # Plot points
        ax.scatter(range(len(target_values)), target_values, 
                 alpha=0.5, color='blue', label='Target', s=5)
        ax.scatter(range(len(pred_values)), pred_values, 
                 alpha=0.5, color='red', label='Prediction', s=5)
        
        # Add title and labels with scientific notation for MSE
        title = f'ID {id}: MSE={avg_mse:.2e}, Pred Ripple={avg_pred_ripple:.1f}%'
        if pred_snr is not None:
            title += f', SNR={pred_snr:.1f} dB'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Pixel Index', fontsize=8)
        ax.set_ylabel('Magnitude', fontsize=8)
        ax.legend(fontsize='small')
        ax.grid(True)
    
    # Clear any unused subplots
    for i in range(n_plot_ids, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_compare_by_id.png", dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}_scatter")

    plt.show()
    
    # Only create the comparison plot if we have valid data
    if not metrics:
        return metrics
        
    # Create a combined comparison plot
    plt.figure(figsize=(fig_size[0] * 2, fig_size[1] * 2))
    
    # Track min/max for identity line
    all_min = float('inf')
    all_max = float('-inf')
    
    # Only plot the IDs we're interested in
    for id in plot_ids:
        mask = (seg == id)
        if np.sum(mask) == 0 or id not in metrics:
            continue
            
        target_values = np.abs(target_sample[mask])
        pred_values = np.abs(prediction_sample[mask])
        
        # Update min/max values
        all_min = min(all_min, np.min(target_values), np.min(pred_values))
        all_max = max(all_max, np.max(target_values), np.max(pred_values))
        
        # Sample points if there are too many (for better visualization)
        max_points = 1000
        if len(target_values) > max_points:
            random_indices = np.random.choice(len(target_values), max_points, replace=False)
            sampled_target = target_values[random_indices]
            sampled_pred = pred_values[random_indices]
        else:
            sampled_target = target_values
            sampled_pred = pred_values
            
        plt.scatter(sampled_target, sampled_pred, alpha=0.5, label=f'ID: {id}', s=5)
    
    # Add identity line
    if all_min != float('inf') and all_max != float('-inf'):
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.5)
    
    plt.title('Target vs Prediction')
    plt.xlabel('Target Magnitude')
    plt.ylabel('Prediction Magnitude')
    plt.legend(fontsize='small')
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}_compare_all.png", dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()
    
    # Print metrics summary
    print("\nMetrics by Segment ID (averaged across all samples):")
    for id, metric in metrics.items():
        output = f"ID {id}: MSE={metric['mse']:.2e}, MAE={metric['mae']:.2e}, "
        
        # Prediction metrics
        output += f"Pred Min={metric['pred_min']:.4f}, Pred Max={metric['pred_max']:.4f}, "
        output += f"Pred Ripple={metric['pred_ripple']:.2f}%"
        if metric['pred_snr'] is not None:
            output += f", Pred SNR={metric['pred_snr']:.1f} dB"
        
        # Target metrics    
        output += f", Target Min={metric['target_min']:.4f}, Target Max={metric['target_max']:.4f}, "
        output += f"Target Ripple={metric['target_ripple']:.2f}%"
        if metric['target_snr'] is not None:
            output += f", Target SNR={metric['target_snr']:.1f} dB"
            
        output += f", Samples={metric['samples']}, Pixels={metric['pixels']}"
        print(output)
    
    return metrics