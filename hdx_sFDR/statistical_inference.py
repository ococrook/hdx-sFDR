import numpy as np
from Bio import PDB
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

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


def kmeans_optimal(data, k_range=range(2, 11), random_state=42):
    """
    Performs k-means clustering with multiple k values and returns the optimal clustering.
    
    Parameters:
    -----------
    data : array-like
        Input matrix to cluster
    k_range : range or list, default=range(2, 11)
        Range of k values to try
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    labels : ndarray
        Cluster labels using the optimal k
    best_k : int
        Optimal number of clusters
    """
    
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    # Initialize storage for metrics
    inertia = []
    silhouette_scores = []
    models = {}
    
    # Try different k values
    for k in k_range:
        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data_array)
        
        # Store the model
        models[k] = kmeans
        
        # Calculate inertia
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score (if k > 1)
        if k > 1:
            score = silhouette_score(data_array, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Find optimal k - silhouette method (best separated clusters)
    if max(silhouette_scores) > 0:
        best_k = list(k_range)[np.argmax(silhouette_scores)]
    else:
        # Fallback to elbow method (look for significant drop in inertia)
        diffs = np.diff(inertia)
        best_k = list(k_range)[np.argmax(-diffs)]
    
    # Return the best labels and k
    best_model = models[best_k]
    return best_model.labels_, best_k

def compute_tst(pvalues, clusters, alpha=0.05):
    """
    Compute the Two-Stage Testing (TST) estimator for proportion of true null hypotheses
    for each cluster, as described in the Adaptive TST GBH procedure.
    
    Parameters:
    -----------
    pvalues : array-like
        List or array of p-values
    clusters : array-like
        Cluster assignments for each p-value
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    tst_estimators : dict
        Dictionary with cluster IDs as keys and TST estimators as values
    rejection_counts : dict
        Dictionary with cluster IDs as keys and number of rejections as values
    """
    
    # Convert inputs to numpy arrays
    pvalues = np.array(pvalues)
    clusters = np.array(clusters)
    
    # Get unique cluster IDs
    unique_clusters = np.unique(clusters)
    
    # Initialize the results dictionaries
    tst_estimators = {}
    rejection_counts = {}
    
    # Compute the modified alpha for step 1
    alpha_prime = alpha / (1 + alpha)
    
    # For each cluster, compute the TST estimator
    for g in unique_clusters:
        # Get p-values for this cluster
        cluster_pvalues = pvalues[clusters == g]
        
        # Get total number of hypotheses in this cluster
        n_g = len(cluster_pvalues)

        # Ensure reweighted_pvalues is 1D
        cluster_pvalues = np.asarray(cluster_pvalues).flatten()
        
        # Step 1: Apply BH procedure at level alpha_prime
        reject_bh, _, _, _ = multipletests(cluster_pvalues, alpha=alpha_prime, method='fdr_bh')
        
        # Count the number of rejections
        r_g1 = np.sum(reject_bh)
        rejection_counts[g] = r_g1
        
        # Step 2: Compute the TST estimator of π_g,0
        gamma_tst_g = (n_g - r_g1) / n_g
        tst_estimators[g] = gamma_tst_g
    
    return tst_estimators, rejection_counts

def adaptive_tst_gbh(pvalues, clusters, alpha=0.05):
    """
    Perform the Adaptive TST GBH procedure on clustered p-values.
    
    Parameters:
    -----------
    pvalues : array-like
        List or array of p-values
    clusters : array-like
        Cluster assignments for each p-value
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    reject : ndarray
        Boolean array indicating which hypotheses are rejected
    reweighted_pvalues : ndarray
        Array of reweighted p-values
    tst_estimators : dict
        Dictionary with cluster IDs as keys and TST estimators as values
    """
    
    # Step 1 & 2: Compute TST estimators
    tst_estimators, rejection_counts = compute_tst(pvalues, clusters, alpha)
    
    # Compute the modified alpha
    alpha_prime = alpha / (1 + alpha)
    
    # Step 3: Reweight p-values using TST estimators
    reweighted_pvalues = compute_weighted_pvalues(pvalues, clusters, tst_estimators)

    # Ensure reweighted_pvalues is 1D
    reweighted_pvalues = np.asarray( reweighted_pvalues).flatten()
    
    # Apply BH procedure on reweighted p-values
    reject, _, _, _ = multipletests(reweighted_pvalues, alpha=alpha_prime, method='fdr_bh')
    
    return reject, reweighted_pvalues, tst_estimators


def compute_weighted_pvalues(pvalues, clusters, tst_estimators):
    """
    Compute reweighted p-values using the TST estimators for each cluster.
    This implements step 3 of the Adaptive TST GBH procedure.
    
    Parameters:
    -----------
    pvalues : array-like
        List or array of p-values
    clusters : array-like
        Cluster assignments for each p-value
    tst_estimators : dict
        Dictionary with cluster IDs as keys and TST estimators as values
        
    Returns:
    --------
    reweighted_pvalues : ndarray
        Array of reweighted p-values
    """
    
    # Convert inputs to numpy arrays
    pvalues = np.array(pvalues)
    clusters = np.array(clusters)
    
    # Initialize the reweighted p-values array
    reweighted_pvalues = np.zeros_like(pvalues)
    
    # For each p-value, apply the reweighting based on its cluster
    for i, (p, cluster) in enumerate(zip(pvalues, clusters)):
        # Get the TST estimator for this cluster
        gamma_tst = tst_estimators[cluster]
        
        # Reweight the p-value
        # We use max(gamma_tst, epsilon) to avoid division by zero
        epsilon = 1e-10
        reweighted_pvalues[i] = p / max(gamma_tst, epsilon)
        
        # Cap the reweighted p-value at 1
        reweighted_pvalues[i] = min(reweighted_pvalues[i], 1.0)
    
    return np.array(reweighted_pvalues)

def compute_qvalues_tst(pvalues, clusters, alpha=0.05):
    """
    Compute q-values based on the Adaptive TST GBH procedure.
    
    Parameters:
    -----------
    pvalues : array-like
        List or array of p-values
    clusters : array-like
        Cluster assignments for each p-value
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    qvalues : ndarray
        Array of q-values corresponding to each p-value
    tst_estimators : dict
        Dictionary with cluster IDs as keys and TST estimators as values
    reweighted_pvalues : ndarray
        Array of reweighted p-values
    """
    
    # First get the TST estimators and reweighted p-values
    _, reweighted_pvalues, tst_estimators = adaptive_tst_gbh(pvalues, clusters, alpha)

    # Ensure reweighted_pvalues is 1D
    reweighted_pvalues = np.asarray(reweighted_pvalues).flatten()
    
    # Sort the reweighted p-values and keep track of original indices
    n = len(reweighted_pvalues)
    sorted_indices = np.argsort(reweighted_pvalues)
    sorted_reweighted_pvalues = reweighted_pvalues[sorted_indices]
    
    # Compute q-values based on the reweighted p-values
    qvalues = np.ones(n)
    
    # Start with the largest p-value
    qvalues[sorted_indices[-1]] = sorted_reweighted_pvalues[-1]
    
    # Process remaining p-values from second-largest to smallest
    for i in range(n-2, -1, -1):
        idx = sorted_indices[i]
        # Q-value is the minimum of the current reweighted p-value and the previous q-value
        qvalues[idx] = min(sorted_reweighted_pvalues[i], qvalues[sorted_indices[i+1]])
    
    return np.real(qvalues), tst_estimators, reweighted_pvalues


def calculate_weighted_pvalues_with_timepoints(peptides, weights, alpha=0.05):
    """
    Calculate weighted p-values and q-values handling timepoints separately but computing
    global q-values across all timepoints.
    
    Parameters:
    peptides (numpy.ndarray): Array containing peptide data with potential timepoints in the 4th column
    weights (numpy.ndarray): Weight matrix for peptides
    alpha: significance level
    
    Returns:
    tuple: (weighted_p_values, weighted_q_values, tst_estimators)
    """
    # Check if timepoints exist (4th column is present)
    has_timepoints = peptides.shape[1] >= 4
    
    if not has_timepoints:
        # Use the original function if no timepoints
        return compute_qvalues_tst(peptides[:,2], weights)
    
    # Extract timepoints and get unique values
    timepoints = peptides[:, 3]
    unique_timepoints = np.unique(timepoints)
    n_timepoints = len(unique_timepoints)
    
    # Compute groupings/clusters
    cluster_labels, best_k = kmeans_optimal(weights)
    print(f"Optimal number of clusters: {best_k}")
    
    # Initialize lists to store results by timepoint
    pvalues_by_timepoint = []
    indices_by_timepoint = []
    tst_estimators_by_timepoint = []
    reweighted_pvalues_by_timepoint = []
    
    # Process each timepoint separately
    for timepoint in unique_timepoints:
        # Get indices for current timepoint
        indices = np.where(timepoints == timepoint)[0]
        indices_by_timepoint.append(indices)
        
        # Extract relevant peptides and their p-values for this timepoint
        current_peptides = peptides[indices]
        current_pvalues = current_peptides[:, 2]
        current_clusters = cluster_labels[indices]
        
        # Store p-values for this timepoint
        pvalues_by_timepoint.append(current_pvalues)
        
        # Compute TST estimators for this timepoint
        tst_estimators, _ = compute_tst(current_pvalues, current_clusters, alpha)
        tst_estimators_by_timepoint.append(tst_estimators)
        
        # Compute reweighted p-values for this timepoint
        reweighted_pvalues = compute_weighted_pvalues(current_pvalues, current_clusters, tst_estimators)
        reweighted_pvalues_by_timepoint.append(reweighted_pvalues)
    
    # Flatten all reweighted p-values for global q-value computation
    all_reweighted_pvalues = []
    timepoint_mapping = []  # Keep track of which timepoint each p-value belongs to
    local_index_mapping = []  # Keep track of the local index within each timepoint
    
    for t, reweighted_pvalues in enumerate(reweighted_pvalues_by_timepoint):
        all_reweighted_pvalues.extend(reweighted_pvalues)
        timepoint_mapping.extend([t] * len(reweighted_pvalues))
        local_index_mapping.extend(range(len(reweighted_pvalues)))
    
    all_reweighted_pvalues = np.array(all_reweighted_pvalues)
    
    # Sort all reweighted p-values
    sorted_indices = np.argsort(all_reweighted_pvalues)
    sorted_reweighted_pvalues = all_reweighted_pvalues[sorted_indices]
    
    # Compute global q-values
    n_total = len(all_reweighted_pvalues)
    global_qvalues_flat = np.ones(n_total)
    
    # Start with the largest p-value
    global_qvalues_flat[sorted_indices[-1]] = sorted_reweighted_pvalues[-1]
    
    # Process remaining p-values from second-largest to smallest
    for i in range(n_total-2, -1, -1):
        idx = sorted_indices[i]
        global_qvalues_flat[idx] = min(sorted_reweighted_pvalues[i], global_qvalues_flat[sorted_indices[i+1]])
    
    # Initialize arrays to store final results in original peptide order
    weighted_p_values = np.ones(len(peptides))
    weighted_q_values = np.ones(len(peptides))
    
    # Map results back to original peptide order
    for t, indices in enumerate(indices_by_timepoint):
        timepoint_reweighted_pvalues = reweighted_pvalues_by_timepoint[t]
        
        # Extract q-values for this timepoint
        timepoint_qvalues_indices = np.where(np.array(timepoint_mapping) == t)[0]
        timepoint_qvalues = global_qvalues_flat[timepoint_qvalues_indices]
        
        # Assign to original peptide array
        weighted_p_values[indices] = timepoint_reweighted_pvalues
        weighted_q_values[indices] = timepoint_qvalues
    
    return weighted_p_values, np.real(weighted_q_values)


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