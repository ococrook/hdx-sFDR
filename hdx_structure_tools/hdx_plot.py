import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_peptide_coverage(processed_data, figsize=(12, 3)):
    """
    Plot peptide coverage map
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    figsize : tuple
        Figure size (width, height)
    """
    peptide_map = processed_data['peptide_map']
    protein_length = len(processed_data['protein_sequence'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each peptide as a line
    for idx, pep in peptide_map.items():
        ax.plot([pep['start'], pep['end']], [idx, idx], 'b-', linewidth=2)
    
    # Add prolines
    sequence = processed_data['protein_sequence']
    pro_positions = [i+1 for i, aa in enumerate(sequence) if aa == 'P']
    ax.vlines(pro_positions, -1, len(peptide_map), 'r', alpha=0.3, label='Prolines')
    
    # Customize plot
    ax.set_xlim(0, protein_length + 1)
    ax.set_ylim(-1, len(peptide_map) + 1)
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Peptide Index')
    ax.set_title('Peptide Coverage Map')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_replicate_correlation(processed_data, figsize=(10, 10)):
    """
    Plot correlation between replicates
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    figsize : tuple
        Figure size (width, height)
    """
    uptake_data = processed_data['uptake_data']
    
    # Get all replicate pairs
    n_replicates = len(uptake_data['replicates'])
    if n_replicates < 2:
        print("Need at least 2 replicates for correlation plot")
        return None
        
    # Print debug info
    print(f"Number of replicates: {n_replicates}")
    for i, rep in enumerate(uptake_data['replicates']):
        print(f"Replicate {i+1} shape: {rep.shape}")
        print(f"Non-zero values: {np.count_nonzero(rep)}")
        print(f"Contains NaN: {np.isnan(rep).any()}")
    
    # Create figure
    fig, axes = plt.subplots(n_replicates-1, n_replicates-1, 
                            figsize=figsize, squeeze=False)
    
    # Plot correlations
    for i in range(n_replicates-1):
        for j in range(i+1, n_replicates):
            # Get data
            rep1 = uptake_data['replicates'][i].flatten()
            rep2 = uptake_data['replicates'][j].flatten()
            
            # Remove pairs where either value is zero
            mask = (rep1 != 0) & (rep2 != 0)
            
            # Calculate correlation if we have data
            corr = np.corrcoef(rep1, rep2)[0,1]

            print(f"Correlation {i+1} vs {j+1}: {corr}")
            print(f"Number of non-zero pairs: {len(rep1)}")
            
            # Plot
            ax = axes[i][j-1]
            ax.scatter(rep1, rep2, alpha=0.5)
            ax.plot([0, max(rep1)], [0, max(rep1)], 'r--', alpha=0.5)
            ax.set_xlabel(f'Replicate {i+1}')
            ax.set_ylabel(f'Replicate {j+1}')
            ax.set_title(f'r = {corr:.3f}')
    
    plt.tight_layout()
    return fig

def plot_deuteration_heatmap(processed_data, figsize=(10, 12), cmap='viridis'):
    """
    Create a heatmap of deuteration levels for all peptides across timepoints
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap to use
    """
    # Get data
    uptake_data = processed_data['uptake_data']
    peptide_map = processed_data['peptide_map']
    timepoints = processed_data['timepoints']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(uptake_data['mean'], aspect='auto', cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Deuterium Uptake')
    
    # Customize axes
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels([f"{t}s" for t in timepoints], rotation=45)
    
    # Create peptide labels
    peptide_labels = [
        f"{pep['start']}-{pep['end']}\n{pep['sequence'][:10]}..." 
        for pep in peptide_map.values()
    ]
    ax.set_yticks(range(len(peptide_labels)))
    ax.set_yticklabels(peptide_labels)
    
    # Add title and labels
    ax.set_title('HDX-MS Deuteration Heatmap')
    ax.set_xlabel('Time')
    ax.set_ylabel('Peptide')
    
    # Adjust layout
    plt.tight_layout()
    plt.close()
    return fig

def plot_peptide_grid(processed_data, max_peptides=20, figsize=(15, 8)):
    """
    Create a grid of deuteration patterns for individual peptides
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    max_peptides : int
        Maximum number of peptides to show
    figsize : tuple
        Figure size (width, height)
    """
    # Get data
    uptake_data = processed_data['uptake_data']
    peptide_map = processed_data['peptide_map']
    timepoints = processed_data['timepoints']
    
    # Select peptides to show
    n_peptides = min(max_peptides, len(peptide_map))
    peptide_indices = np.linspace(0, len(peptide_map)-1, n_peptides, dtype=int)
    
    # Calculate grid dimensions
    n_cols = 4
    n_rows = (n_peptides + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each peptide
    for i, pep_idx in enumerate(peptide_indices):
        ax = axes[i]
        pep = peptide_map[pep_idx]
        
        # Plot uptake curve with error bars
        ax.errorbar(timepoints, 
                   uptake_data['mean'][pep_idx], 
                   yerr=uptake_data['std'][pep_idx],
                   marker='o')
        
        # Add title and labels
        ax.set_title(f"{pep['start']}-{pep['end']}\n{pep['sequence'][:10]}...")
        ax.set_xscale('log')
        
        # Only add labels for edge plots
        if i >= len(peptide_indices) - n_cols:
            ax.set_xlabel('Time (s)')
        if i % n_cols == 0:
            ax.set_ylabel('Uptake')
    
    # Remove empty subplots
    for i in range(len(peptide_indices), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def plot_residue_heatmap(processed_data, figsize=(12, 6)):
    """
    Create a residue-level heatmap using peptide overlap information
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    figsize : tuple
        Figure size (width, height)
    """
    # Get data
    peptide_matrix = processed_data['peptide_matrix']
    uptake_data = processed_data['uptake_data']
    timepoints = processed_data['timepoints']
    protein_sequence = processed_data['protein_sequence']
    
    # Initialize residue-level deuteration matrix
    n_residues = len(protein_sequence)
    n_timepoints = len(timepoints)
    residue_deuteration = np.zeros((n_residues, n_timepoints))
    coverage_count = np.zeros((n_residues, n_timepoints))
    
    # For each timepoint
    for t in range(n_timepoints):
        # For each peptide
        for p in range(peptide_matrix.shape[0]):
            # Get peptide coverage and deuteration
            peptide_residues = peptide_matrix[p]
            deuteration = uptake_data['mean'][p, t]
            
            # Assuming equal distribution across exchangeable residues
            n_covered = peptide_residues.sum()
            if n_covered > 0:  # Avoid division by zero
                deut_per_residue = deuteration / n_covered
                
                # Add to residue matrix
                residue_deuteration[:, t] += peptide_residues * deut_per_residue
                coverage_count[:, t] += peptide_residues
    
    # Average by coverage count (avoiding division by zero)
    mask = coverage_count > 0
    residue_deuteration[mask] = residue_deuteration[mask] / coverage_count[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot deuteration heatmap
    im = ax.imshow(residue_deuteration.T, aspect='auto', cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Deuteration per Residue')

    # Customize axes
    ax.set_yticks(range(len(timepoints)))
    ax.set_yticklabels([f"{t}s" for t in timepoints])

    # Add residue numbers every 10 positions
    xticks = range(0, n_residues, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    # Add title and labels
    ax.set_title('Residue-Level Deuteration Heatmap')
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Time')
        
    # Adjust layout
    plt.tight_layout()
    return fig, residue_deuteration

def plot_coverage_uptake(processed_data, timepoint_idx=None, figsize=(12, 6), cmap='viridis'):
    """
    Create a peptide coverage plot with uptake values shown by color
    
    Parameters:
    -----------
    processed_data : dict
        Output from process_hdx_dataset
    timepoint_idx : int or None
        Index of timepoint to show. If None, creates subplots for all timepoints
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap to use for uptake values
    """
    # Get data
    peptide_matrix = processed_data['peptide_matrix']
    peptide_map = processed_data['peptide_map']
    uptake_data = processed_data['uptake_data']
    timepoints = processed_data['timepoints']
    protein_sequence = processed_data['protein_sequence']

    vmax = uptake_data['mean'].max()
    
    if timepoint_idx is not None:
        # Single timepoint plot
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        timepoint_indices = [timepoint_idx]
        fig.suptitle(f'Peptide Coverage and Uptake at {timepoints[timepoint_idx]}s')
        n_cols = 1
    else:
        # Multiple timepoint plots
        n_timepoints = len(timepoints)
        n_cols = min(3, n_timepoints)
        n_rows = (n_timepoints + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows/2))
        if n_rows == 1:
            axes = [axes] if n_timepoints == 1 else axes.flatten()
        else:
            axes = axes.flatten()
        timepoint_indices = range(n_timepoints)
        fig.suptitle('Peptide Coverage and Uptake Over Time')
    
    # Create colormap
    cmap = plt.get_cmap(cmap)
    
    # Plot each timepoint
    for ax_idx, t_idx in enumerate(timepoint_indices):
        ax = axes[ax_idx]
        
        # Get uptake values for this timepoint
        uptakes = uptake_data['mean'][:, t_idx]
        
        # Plot each peptide as a colored rectangle
        for pep_idx in range(peptide_matrix.shape[0]):
            peptide = peptide_map[pep_idx]
            start = peptide['start'] - 1  # Convert to 0-based indexing
            end = peptide['end']
            uptake = uptakes[pep_idx]
            
            # Create rectangle
            rect = plt.Rectangle((start, pep_idx-0.4), end-start, 0.8,
                                   facecolor=cmap(uptake/vmax),
                                   alpha=0.7)
            ax.add_patch(rect)
        
        # Customize axes
        ax.set_xlim(-1, len(protein_sequence)+1)
        ax.set_ylim(-1, peptide_matrix.shape[0])
        
        # Add labels
        if ax_idx >= len(axes) - n_cols:  # Bottom row
            ax.set_xlabel('Residue Position')
        if ax_idx % n_cols == 0:  # Leftmost column
            ax.set_ylabel('Peptide Index')
            
        # Add title for each subplot
        ax.set_title(f't = {timepoints[t_idx]}s')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add tick marks
        tick_spacing = 10
        ax.set_xticks(range(0, len(protein_sequence), tick_spacing))
    
    # Remove empty subplots
    if timepoint_idx is None:
        for idx in range(len(timepoints), len(axes)):
            fig.delaxes(axes[idx])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Deuterium Uptake')

    return fig

def plot_predicted_deuteration_heatmap(residue_deuteration, timepoints, figsize=(10, 6)):
    """
    Plot a heatmap of predicted deuteration values
    
    Parameters:
    residue_deuteration: 2D numpy array (residues x timepoints) of predicted deuteration values
    timepoints: List/array of timepoint values in seconds
    figsize: Tuple for figure size (width, height)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot deuteration heatmap
    im = ax.imshow(residue_deuteration.T, aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Deuteration')
    
    # Customize axes
    ax.set_yticks(range(len(timepoints)))
    ax.set_yticklabels([f"{t}s" for t in timepoints])
    
    # Add residue numbers every 10 positions
    n_residues = residue_deuteration.shape[0]
    xticks = range(0, n_residues, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    
    # Add title and labels
    ax.set_title('Predicted Residue-Level Deuteration')
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Time')
    
    plt.tight_layout()
    return fig

# Function to plot loss history
def plot_loss_history(loss_history, figsize=(12, 6)):
    """
    Plot the evolution of all loss components during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot total loss
    ax1.plot(loss_history['total'], label='Total Loss')
    ax1.set_title('Total Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    
    # Plot individual components
    components = ['peptide', 'sequence', 'spatial', 'monotonicity']
    for component in components:
        ax2.plot(loss_history[component], label=component.capitalize())
    ax2.set_title('Loss Components vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_predicted_coverage_uptake(peptide_matrix, predicted_deuteration, experimental_uptake, 
                                 peptide_map, protein_sequence, timepoints, 
                                 timepoint_idx=None, figsize=(15, 6), cmap='viridis'):
    """
    Create a peptide coverage plot comparing predicted and experimental uptake values
    
    Parameters:
    -----------
    peptide_matrix : torch.Tensor or numpy.array
        Binary matrix mapping peptides to residues
    predicted_deuteration : torch.Tensor or numpy.array
        Predicted residue-level deuteration values
    experimental_uptake : torch.Tensor or numpy.array
        Experimental peptide-level uptake values
    peptide_map : list of dict
        List of peptide information (start, end positions)
    protein_sequence : str
        Protein sequence
    timepoints : list
        List of timepoints
    timepoint_idx : int or None
        Index of timepoint to show. If None, shows all timepoints
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(peptide_matrix):
        peptide_matrix = peptide_matrix.numpy()
    if torch.is_tensor(predicted_deuteration):
        predicted_deuteration = predicted_deuteration.numpy()
    if torch.is_tensor(experimental_uptake):
        experimental_uptake = experimental_uptake.numpy()
    
    # Calculate predicted peptide uptakes
    predicted_uptake = np.matmul(peptide_matrix, predicted_deuteration)
    
    if timepoint_idx is not None:
        # Single timepoint plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [[ax1], [ax2]]
        timepoint_indices = [timepoint_idx]
        fig.suptitle(f'Peptide Coverage and Uptake at {timepoints[timepoint_idx]}s')
    else:
        # Multiple timepoint plots
        n_timepoints = len(timepoints)
        n_rows = (n_timepoints + 2) // 3
        fig, axes = plt.subplots(n_rows, 6, figsize=(figsize[0]*2, figsize[1]*n_rows/2))
        if n_rows == 1:
            axes = [axes[:3], axes[3:]]
        timepoint_indices = range(n_timepoints)
        fig.suptitle('Peptide Coverage and Uptake Over Time')
    
    titles = ['Predicted Uptake', 'Experimental Uptake']
    uptakes = [predicted_uptake, experimental_uptake]
    
    # Create colormap
    cmap = plt.get_cmap(cmap)
    vmax = max(predicted_uptake.max(), experimental_uptake.max())
    
    # Plot each timepoint
    for row_idx, (title, uptake_data) in enumerate(zip(titles, uptakes)):
        for ax_idx, t_idx in enumerate(timepoint_indices):
            ax = axes[row_idx][ax_idx]
            
            # Get uptake values for this timepoint
            uptakes = uptake_data[:, t_idx]
            
            # Plot each peptide as a colored rectangle
            for pep_idx in range(peptide_matrix.shape[0]):
                peptide = peptide_map[pep_idx]
                start = peptide['start'] - 1
                end = peptide['end']
                uptake = uptakes[pep_idx]
                
                rect = plt.Rectangle((start, pep_idx-0.4), end-start, 0.8,
                                   facecolor=cmap(uptake/vmax),
                                   alpha=0.7)
                ax.add_patch(rect)
            
            # Customize axes
            ax.set_xlim(-1, len(protein_sequence)+1)
            ax.set_ylim(-1, peptide_matrix.shape[0])
            
            if ax_idx >= len(axes[0]) - 3:  # Bottom row
                ax.set_xlabel('Residue Position')
            if ax_idx % 3 == 0:  # Leftmost column
                ax.set_ylabel('Peptide Index')
            
            ax.set_title(f'{title}\nt = {timepoints[t_idx]}s')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add tick marks
            tick_spacing = 10
            ax.set_xticks(range(0, len(protein_sequence), tick_spacing))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Deuterium Uptake')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    return fig

def plot_uptake_comparison(peptide_matrix, predicted_deuteration, experimental_uptake, 
                          peptide_map, protein_sequence, timepoints, 
                          timepoint_idx=None, figsize=(12, 6), cmap='RdBu_r'):
    """
    Create a peptide coverage plot showing the difference between predicted and experimental uptake
    
    Parameters:
    -----------
    peptide_matrix : torch.Tensor or numpy.array
        Binary matrix mapping peptides to residues
    predicted_deuteration : torch.Tensor or numpy.array
        Predicted residue-level deuteration values
    experimental_uptake : torch.Tensor or numpy.array
        Experimental peptide-level uptake values
    peptide_map : list of dict
        List of peptide information (start, end positions)
    protein_sequence : str
        Protein sequence
    timepoints : list
        List of timepoints
    timepoint_idx : int or None
        Index of timepoint to show. If None, shows all timepoints
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(peptide_matrix):
        peptide_matrix = peptide_matrix.numpy()
    if torch.is_tensor(predicted_deuteration):
        predicted_deuteration = predicted_deuteration.numpy()
    if torch.is_tensor(experimental_uptake):
        experimental_uptake = experimental_uptake.numpy()
    
    # Calculate predicted peptide uptakes
    predicted_uptake = np.matmul(peptide_matrix, predicted_deuteration)
    
    # Calculate differences
    differences = predicted_uptake - experimental_uptake
    
    if timepoint_idx is not None:
        # Single timepoint plot
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        timepoint_indices = [timepoint_idx]
        fig.suptitle(f'Difference in Predicted vs Experimental Uptake at {timepoints[timepoint_idx]}s')
        n_cols = 1
    else:
        # Multiple timepoint plots
        n_timepoints = len(timepoints)
        n_cols = min(3, n_timepoints)
        n_rows = (n_timepoints + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows/2))
        if n_rows == 1:
            axes = [axes] if n_timepoints == 1 else axes.flatten()
        else:
            axes = axes.flatten()
        timepoint_indices = range(n_timepoints)
        fig.suptitle('Difference in Predicted vs Experimental Uptake Over Time')
    
    # Create diverging colormap centered at 0
    cmap = plt.get_cmap(cmap)
    vmax = max(abs(differences.min()), abs(differences.max()))
    norm = plt.Normalize(-vmax, vmax)
    
    # Plot each timepoint
    for ax_idx, t_idx in enumerate(timepoint_indices):
        ax = axes[ax_idx]
        
        # Get difference values for this timepoint
        diffs = differences[:, t_idx]
        
        # Plot each peptide as a colored rectangle
        for pep_idx in range(peptide_matrix.shape[0]):
            peptide = peptide_map[pep_idx]
            start = peptide['start'] - 1
            end = peptide['end']
            diff = diffs[pep_idx]
            
            rect = plt.Rectangle((start, pep_idx-0.4), end-start, 0.8,
                               facecolor=cmap(norm(diff)),
                               alpha=0.7,
                               edgecolor='black',
                               linewidth = 0.2)
            ax.add_patch(rect)
        
        # Customize axes
        ax.set_xlim(-1, len(protein_sequence)+1)
        ax.set_ylim(-1, peptide_matrix.shape[0])
        
        if ax_idx >= len(axes) - n_cols:  # Bottom row
            ax.set_xlabel('Residue Position')
        if ax_idx % n_cols == 0:  # Leftmost column
            ax.set_ylabel('Peptide Index')
        
        ax.set_title(f't = {timepoints[t_idx]}s')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add tick marks
        tick_spacing = 10
        ax.set_xticks(range(0, len(protein_sequence), tick_spacing))
    
    # Remove empty subplots
    if timepoint_idx is None:
        for idx in range(len(timepoints), len(axes)):
            fig.delaxes(axes[idx])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Predicted - Experimental\nDeuterium Uptake')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    return fig