import numpy as np
from Bio import PDB
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

def load_structure(cif_path):
    """Load CIF file and extract coordinates and B-factors (pLDDT scores for AlphaFold)"""
    # Load structure
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('protein', cif_path)
    model = structure[0]
    
    # Get coordinates and pLDDT scores
    coords = []
    plddt = []
    residue_ids = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get CA atom coordinates
                ca_atom = residue['CA']
                coords.append(ca_atom.get_coord())
                # B-factor contains pLDDT score in AlphaFold
                plddt.append(ca_atom.get_bfactor())
                residue_ids.append(residue.id[1])
    
    return np.array(coords), np.array(plddt), np.array(residue_ids)

def process_structure(structure):
    """process structure and extract coordinates and B-factors (pLDDT scores for AlphaFold)"""
    # Load structure
    model = structure[0]
    
    # Get coordinates and pLDDT scores
    coords = []
    plddt = []
    residue_ids = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get CA atom coordinates
                ca_atom = residue['CA']
                coords.append(ca_atom.get_coord())
                # B-factor contains pLDDT score in AlphaFold
                plddt.append(ca_atom.get_bfactor())
                residue_ids.append(residue.id[1])
    
    return np.array(coords), np.array(plddt), np.array(residue_ids)




def process_peptide_map(peptide_df):
    """Process peptide map dataframe
    Expected columns: start, end, pvalue, exposure (optional)"""

    # Check if exposure column is present
    if 'Exposure' in peptide_df.columns:
        # If exposure column is present, include it in the output array
        peptides = peptide_df[['Start', 'End', 'pvalue', 'Exposure']].values
    else:
        # If exposure column is not present, proceed as before
        peptides = peptide_df[['Start', 'End', 'pvalue']].values

    return peptides

def calculate_peptide_centroids(peptides, coordinates):
    """Calculate centroid coordinates for each peptide"""
    centroids = []
    for start, end, _ in peptides:
        # Get coordinates for residues in peptide
        # Ensure start and end are integers for slicing
        start = int(start)  # Convert start to integer
        end = int(end)  # Convert end to integer

        peptide_coords = coordinates[start-1:end] 
        centroid = np.mean(peptide_coords, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def calculate_weights(peptides, coordinates, plddt_scores, lambda_seq=10, lambda_struct=15, alpha = 0.5):

    
    """
    Calculate sequence-based weights accounting for peptide overlap
    
    Parameters:
    -----------
    peptides : array-like
        Array of peptides with columns [start, end, pvalue]
    lambda_seq : float
        Length scale for sequence distance decay
    
    Returns:
    --------
    array
        Matrix of sequence-based weights
    """

    # check if exposure column is present
    if np.shape(peptides)[1] == 4:
      
      # get the value in this column
      first_value = peptides[0, 3]

      # Filter the array
      peptides = peptides[peptides[:, 3] == first_value]

      # Remove the last column
      peptides = peptides[:, :-1]




    n_peptides = len(peptides)
    seq_weights = np.zeros((n_peptides, n_peptides))
    
    for i in range(n_peptides):
        for j in range(n_peptides):
            # Calculate midpoint distance
            mid_i = (peptides[i,0] + peptides[i,1]) / 2
            mid_j = (peptides[j,0] + peptides[j,1]) / 2
            seq_dist = abs(mid_i - mid_j)
            
            # Calculate overlap
            start_i, end_i = peptides[i,0], peptides[i,1]
            start_j, end_j = peptides[j,0], peptides[j,1]
            
            overlap = max(0, min(end_i, end_j) - max(start_i, start_j))
            
            # Adjust weight based on overlap
            if overlap > 0:
                # Calculate fraction of overlap
                len_i = end_i - start_i
                len_j = end_j - start_j
                overlap_frac = overlap / min(len_i, len_j)
                seq_dist = seq_dist * (1 - overlap_frac)
                
                # Increase weight for overlapping peptides
                seq_weights[i,j] = np.exp(-seq_dist / lambda_seq)
            else:
                # Standard distance-based weight for non-overlapping peptides
                seq_weights[i,j] = np.exp(-seq_dist / lambda_seq)
    
    
    # Calculate structure-based weights
    centroids = calculate_peptide_centroids(peptides, coordinates)
    struct_dist = squareform(pdist(centroids))
    struct_weights = np.exp(-struct_dist / lambda_struct)
    
    # Calculate confidence weights
    conf_weights = np.zeros((n_peptides, n_peptides))
    for i in range(n_peptides):
        for j in range(n_peptides):
            plddt_i = np.mean(plddt_scores[int(peptides[i,0]-1):int(peptides[i,1])])
            plddt_j = np.mean(plddt_scores[int(peptides[j,0]-1):int(peptides[j,1])])
            conf_weights[i,j] = np.sqrt(plddt_i * plddt_j) / 100
    
    # Combine weights
    weights = alpha*seq_weights  +  ((1 - alpha)*struct_weights * conf_weights)
    
    return weights, seq_weights, struct_weights, conf_weights

def calculate_weighted_pvalues(peptides, weights, transform_sum=True):
    """Calculate weighted p-values using either approach"""
    n_peptides = len(peptides)
    pvalues = peptides[:,2]
    
    # Normalize weights
    w_norm = weights / weights.sum(axis=1)[:,None]

    # Adjust extreme p-values to prevent infinite z-scores
    # Set a small epsilon value
    epsilon = 1e-10
    
    # Clip p-values to be within (epsilon, 1-epsilon)
    pvalues = np.clip(pvalues, epsilon, 1-epsilon)
    
    if transform_sum:
        # Approach 1: Transform of sum
        z_scores = norm.ppf(pvalues)
        weighted_z = np.sum(w_norm * z_scores[None,:], axis=1)
        weighted_p = norm.cdf(weighted_z)
    else:
        # Approach 2: Sum of transforms
        z_scores = norm.ppf(pvalues)
        weighted_p = np.sum(w_norm * pvalues[None, :], axis=1)


    meff, eigenvalues, corr_matrix = estimate_effective_tests_from_weights(weights)
    weighted_q_values = compute_qvalues(weighted_p, meff)

    return weighted_p, weighted_q_values

def calculate_weighted_pvalues_with_timepoints(peptides, weights, transform_sum=True):
    """
    Calculate weighted p-values using either approach, handling timepoints separately
    
    Parameters:
    peptides (numpy.ndarray): Array containing peptide data with potential timepoints in the 4th column
    weights (numpy.ndarray): Weight matrix for peptides
    transform_sum (bool): Whether to use transform of sum (True) or sum of transforms (False)
    
    Returns:
    tuple: (weighted_p_values, weighted_q_values)
    """
    # Check if timepoints exist (4th column is present)
    has_timepoints = peptides.shape[1] >= 4
    
    if not has_timepoints:
        # Use the original function if no timepoints
        return calculate_weighted_pvalues(peptides, weights, transform_sum)
    
    # Extract timepoints and get unique values
    timepoints = peptides[:, 3]
    unique_timepoints = np.unique(timepoints)
    n_timepoints = len(unique_timepoints)
    
    # Initialize arrays to store results
    all_weighted_p = []

    # Normalize weights for the current set of peptides
    current_weights = weights
    w_norm = current_weights / current_weights.sum(axis=1)[:, None]
    
    # Process each timepoint separately
    for timepoint in unique_timepoints:
        # Get indices for current timepoint
        indices = np.where(timepoints == timepoint)[0]
        
        # Extract relevant peptides and their p-values for this timepoint
        current_peptides = peptides[indices]
        current_pvalues = current_peptides[:, 2]
        
        
        # Adjust extreme p-values to prevent infinite z-scores
        epsilon = 1e-10
        current_pvalues = np.clip(current_pvalues, epsilon, 1-epsilon)
        
        if transform_sum:
            # Approach 1: Transform of sum
            z_scores = norm.ppf(current_pvalues)
            weighted_z = np.sum(w_norm * z_scores[None,:], axis=1)
            weighted_p = norm.cdf(weighted_z)
        else:
            # Approach 2: Sum of transforms
            z_scores = norm.ppf(current_pvalues)
            weighted_p = np.sum(w_norm * current_pvalues[None, :], axis=1)
        
        all_weighted_p.append(weighted_p)
    

    # Need to maintain original order
    # Initialize arrays to store results in original order
    weighted_p = np.zeros(len(peptides))
    # Put results back in the original positions

    start_idx = 0
    for i, timepoint in enumerate(unique_timepoints):
        indices = np.where(timepoints == timepoint)[0]
        weighted_p[indices] = all_weighted_p[i]
    
    # Calculate effective tests once for all data
    meff, eigenvalues, corr_matrix = estimate_effective_tests_from_weights(weights)
    
    # Adjust meff by multiplying with the number of timepoints
    adjusted_meff = meff * n_timepoints
    
    # Compute q-values using the adjusted meff
    weighted_q_values = compute_qvalues(weighted_p, adjusted_meff)
    
    return weighted_p, weighted_q_values


def compute_qvalues(pvalues, meff = None):
    
    n = len(pvalues)
    
    if meff is None:
      meff = n

    # Order p-values (decreasing)
    o = np.argsort(pvalues)[::-1]  # decreasing order
    
    # Calculate cumulative minimum of adjusted p-values
    i = np.arange(n, 0, -1)  # n down to 1
    qvalues = np.minimum(1, np.minimum.accumulate(meff/i * pvalues[o]))
    
    # Return to original order
    ro = np.zeros_like(o)
    ro[o] = np.arange(n)
    qvalues = qvalues[ro]
    
    return qvalues



def prepare_peptide_map(df, effect_positions=None):
    """
    Extract unique peptides from dataframe and add random uniform p-values
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'start' and 'end' columns (possibly repeated)
    
    Returns:
    --------
    pandas DataFrame
        Unique peptides with random p-values
    """
    # Get unique start-end combinations
    unique_peptides = df[['Start', 'End']].drop_duplicates()
    
    # Sort by start position
    unique_peptides = unique_peptides.sort_values('Start').reset_index(drop=True)
    
    # Add random uniform p-values
    np.random.seed(42)  # for reproducibility
    unique_peptides['pvalue'] = np.random.uniform(0, 1, size=len(unique_peptides))
    
    # Insert effects if specified
    if effect_positions:
        for start_res, end_res in effect_positions:
            # Find peptides overlapping with effect region
            mask = ((unique_peptides['Start'] <= end_res) & 
                   (unique_peptides['End'] >= start_res))
            
            # Set low p-values for these peptides
            n_affected = mask.sum()
            if n_affected > 0:
                unique_peptides.loc[mask, 'pvalue'] = np.random.uniform(0, 0.001, 
                                                                      size=n_affected)
    
    return unique_peptides

def estimate_effective_tests_from_weights(weights):
    
    
    # Normalize weights
    w_norm = weights / weights.sum(axis=1)[:,None]
    
    corr_matrix = np.dot(w_norm.T, w_norm)
    
    # Normalize to ensure diagonal is 1
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / d[:, None] / d[None, :]

    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(corr_matrix)
    
    # Li and Ji method
    meff = sum(np.minimum(1, eigenvalues))
    return meff, eigenvalues, corr_matrix 


def main():
    # Example usage
    # Load structure
    coords, plddt, residue_ids = load_structure('protein.pdb')
    
    # Load peptide map
    # Assuming DataFrame with columns: start, end, pvalue
    peptide_df = pd.read_csv('peptide_map.csv')
    peptides = process_peptide_map(peptide_df)
    
    # Calculate weights
    weights = calculate_weights(peptides, coords, plddt)
    
    # Calculate weighted p-values (try both approaches)
    weighted_p1 = calculate_weighted_pvalues(peptides, weights, transform_sum=True)
    weighted_p2 = calculate_weighted_pvalues(peptides, weights, transform_sum=False)
    
    # Output results
    results = pd.DataFrame({
        'start': peptides[:,0],
        'end': peptides[:,1],
        'original_p': peptides[:,2],
        'weighted_p_transform_sum': weighted_p1,
        'weighted_p_sum_transform': weighted_p2
    })
    
    return results

def analyze_hdx_data(df, structure, lambda_seq=1, lambda_struct=1, alpha=0.5, transform_sum=True):
    """
    Analyze HDX-MS data with structural information to correct p-values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing HDX-MS data with required columns (Start, End, pvalue)
    structure : Bio.PDB.Structure.Structure
        Protein structure object
    lambda_seq : float, optional
        Sequence weight parameter. Default is 1.
    lambda_struct : float, optional
        Structure weight parameter. Default is 1.
    alpha : float, optional
        Confidence weight parameter. Default is 0.5.
    transform_sum : bool, optional
        Whether to use sum transformation for weighted p-values. Default is True.
        
    Returns:
    --------
    pandas.DataFrame
        Results containing original and corrected statistics
    """
    
    # Process structure
    print("Processing structure...")
    coords, plddt, residue_ids = process_structure(structure=structure)
    
    # Process peptide data
    print("Processing peptide data...")
    peptides = process_peptide_map(df)
    
    # Calculate weights
    print("Calculating weights...")
    weights, seq_weights, struct_weights, conf_weights = calculate_weights(
        peptides,
        coords,
        plddt,
        lambda_seq=lambda_seq,
        lambda_struct=lambda_struct,
        alpha=alpha
    )
    
    # Calculate weighted p-values and q-values
    print("Calculating corrected statistics...")
    weighted_pvalues, qvalues = calculate_weighted_pvalues_with_timepoints(
        peptides, 
        weights, 
        transform_sum=transform_sum
    )
    
    # Estimate effective number of tests
    meff, eigenvalues, corr_matrix = estimate_effective_tests_from_weights(weights)
    
    # Prepare results
    results = pd.DataFrame({
        'start': peptides[:,0],
        'end': peptides[:,1],
        'original_p': peptides[:,2],
        'original_q': compute_qvalues(peptides[:,2]),
        'corrected_q': compute_qvalues(
            peptides[:,2], 
            meff=meff * len(np.unique(peptides[:, 3]))
        ),
        'weighted_p': weighted_pvalues,
        'weighted_q': qvalues
    })
    
    print(f"Analysis complete! Processed {len(results)} peptides.")
    print(f"Estimated effective number of tests: {meff:.2f}")
    
    return results