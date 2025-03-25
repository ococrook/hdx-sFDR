# Standard library imports
import warnings
from typing import Dict, Union, Any, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats

# BioPython imports
import Bio
from Bio import PDB
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, DSSP
from Bio.PDB.Structure import Structure

"""
Utilities for extracting and analyzing structural features from protein structures.

This module provides functions to process mmCIF files and extract features
such as contact maps, distance matrices
"""

def extract_structural_features(cif_path: str, distance_cutoff: float=8.0) -> Dict[str, Union[np.ndarray, Any]]:
    """
    Extract contact map and distances from CIF file
    
    Parameters:
    -----------
    cif_path : str
        Path to CIF file
    distance_cutoff : float, optional
        Distance cutoff in Angstroms for contacts. Default is 8.0.
        
    Returns:
    --------
    Dict[str, Union[np.ndarray, Structure]]
        Dictionary containing:
        - contact_map: Binary contact map (numpy array)
        - distances: Distance matrix in Angstroms (numpy array)
        - structure: Loaded structure object (Bio.PDB.Structure)
    """
    # Load structure
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('protein', cif_path)
    model = structure[0]
    
    # Get list of residues
    residues = [res for res in model.get_residues()]
    n_residues = len(residues)
    
    # Initialize distance matrix
    distances = np.zeros((n_residues, n_residues))
    
    # Compute distances between CA atoms
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues):
            try:
                ca1 = res1['CA']
                ca2 = res2['CA']
                distance = ca1 - ca2
                distances[i,j] = distance
            except KeyError:
                distances[i,j] = float('inf')
    
    # Create contact map
    contact_map = distances <= distance_cutoff
    
    return {
        'contact_map': contact_map,
        'distances': distances,
        'structure': structure
    }

def plot_contact_map(features: Dict[str, Any], figsize: Tuple[int, int] = (10, 10)) -> Figure:
    """
     Parameters:
    -----------
    features : Dict[str, Any]
        Dictionary containing structural features.
        Must include 'contact_map' key with a numpy array.
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (10, 10).
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object containing the contact map visualization.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(features['contact_map'], cmap='binary')
    ax.set_title('Contact Map (CA distance <= 8Å)')
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Residue Position')
    
    plt.colorbar(im, ax=ax)
    plt.close()
    return fig

def analyze_contact_hdx_correlation(contact_map, residue_deuteration, timepoint_idx=0):
    """
    Analyze correlation between contact map and HDX residue deuteration
    
    Parameters:
    contact_map: 2D numpy array of contacts (1 for contact, 0 for no contact)
    residue_deuteration: 2D array (residues x timepoints) of average deuteration values
    timepoint_idx: Which timepoint to analyze
    """
    deut_slice = residue_deuteration[:, timepoint_idx]
        
    # Calculate average deuteration difference for residues that are in contact vs not in contact
    in_contact_deut = []
    no_contact_deut = []
    
    for i in range(len(contact_map)):
        for j in range(i+1, len(contact_map)):  # Only look at upper triangle to avoid duplicates
            if contact_map[i,j] == 1:
                in_contact_deut.append(abs(deut_slice[i] - deut_slice[j]))
            else:
                no_contact_deut.append(abs(deut_slice[i] - deut_slice[j]))
    
    # Convert to arrays for easier analysis
    in_contact = np.array(in_contact_deut)
    no_contact = np.array(no_contact_deut)
    
    # Calculate statistics
    contact_mean = np.mean(in_contact)
    no_contact_mean = np.mean(no_contact)
    
    # Perform t-test
    t_stat, p_value = scipy.stats.ttest_ind(in_contact, no_contact)
    
    return {
        'contact_mean_diff': contact_mean,
        'no_contact_mean_diff': no_contact_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'in_contact_diffs': in_contact,
        'no_contact_diffs': no_contact
    }

# Visualization
def plot_contact_hdx_comparison(contact_map, residue_deuteration, timepoint_idx=0, timepoints=None):
    """
    Create a side-by-side visualization of contact map and deuteration differences
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot contact map
    im1 = ax1.imshow(contact_map, cmap='binary')
    ax1.set_title('Contact Map')
    plt.colorbar(im1, ax=ax1)
    
    # Get deuteration at specified timepoint
    deut_slice = residue_deuteration[:, timepoint_idx]
    
    # Create a distance matrix of deuteration differences
    n_res = len(deut_slice)
    deut_diff = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            deut_diff[i,j] = abs(deut_slice[i] - deut_slice[j])
    
    im2 = ax2.imshow(deut_diff, cmap='viridis')
    ax2.set_title(f'Deuteration Difference Matrix (t={timepoints[timepoint_idx]}s)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Add a box plot to compare distributions
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    results = analyze_contact_hdx_correlation(contact_map, residue_deuteration, timepoint_idx)
    
    box_data = [results['in_contact_diffs'], results['no_contact_diffs']]
    ax3.boxplot(box_data, labels=['In Contact', 'Not in Contact'])
    ax3.set_ylabel('Absolute Deuteration Difference')
    ax3.set_title(f'Distribution of Deuteration Differences (t={timepoints[timepoint_idx]}s)')
    
    return fig, fig2