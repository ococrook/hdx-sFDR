import pandas as pd
import numpy as np
import torch

def read_fasta(filepath):
    """
    Read protein sequence from FASTA file
    
    Parameters:
    -----------
    filepath : str
        Path to FASTA file
        
    Returns:
    --------
    dict
        Dictionary with sequence IDs as keys and sequences as values
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if it exists
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                # Start new sequence
                current_id = line[1:].split()[0]  # Get ID without '>' and description
                current_seq = []
            elif line:  # Sequence line
                current_seq.append(line)
    
    # Save last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences

def validate_hdx_csv(df):
    """
    Validate that a DataFrame contains the required columns for HDX-MS analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool
        True if validation passes, False otherwise
    
    Notes:
    ------
    Required columns: 'Start', 'End', 'pvalue'
    Optional columns: 'Exposure'
    """
    # Check for required columns
    required_columns = ['Start', 'End', 'pvalue']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        print(f"Required columns are: {', '.join(required_columns)}")
        print(f"Found columns: {', '.join(df.columns)}")
        return False
    
    # Check for optional columns
    optional_columns = ['Exposure']
    missing_optional = [col for col in optional_columns if col not in df.columns]
    
    if missing_optional:
        print(f"Warning: Missing optional column: {', '.join(missing_optional)}")
        print("This is okay, but some functionality may be limited.")
    
    # Check column data types
    try:
        # Verify Start and End are numeric
        df['Start'].astype(int)
        df['End'].astype(int)
        
        # Verify pvalue is numeric and between 0-1
        p_values = df['pvalue'].astype(float)
        if (p_values < 0).any() or (p_values > 1).any():
            print("Warning: Some p-values are outside the valid range [0,1]")
    
    except ValueError as e:
        print(f"Error: Invalid data types in required columns: {e}")
        return False
        
    print("CSV validation successful!")
    return True

def create_peptide_matrix(df, protein_sequence, protein_length=None):
    """
    Create binary matrix mapping peptides to exchangeable residues
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed HDX data
    protein_sequence : str
        Full protein sequence
    protein_length : int, optional
        Length of protein. If None, inferred from data
        
    Returns:
    --------
    np.ndarray
        Binary matrix of shape (n_peptides, n_residues)
        Each row represents which exchangeable residues contribute to that peptide
    dict
        Mapping of matrix rows to peptide info
    """
    # Get unique peptides and reset index
    peptides = df[['Start', 'End', 'Sequence']].drop_duplicates().reset_index(drop=True)
    
    # Infer protein length if not provided
    if protein_length is None:
        protein_length = len(protein_sequence)
    
    # Create matrix
    matrix = np.zeros((len(peptides), protein_length))
    
    # Fill matrix accounting for biochemical constraints
    for idx in range(len(peptides)):
        peptide = peptides.iloc[idx]
        start = peptide['Start'] - 1  # Convert to 0-based indexing
        end = peptide['End']
        
        for pos in range(start, end):
            # Skip first two residues of protein (fast back-exchange)
            if pos < 2:
                continue
                
            # Skip prolines (no exchangeable amide)
            if protein_sequence[pos] == 'P':
                continue
                
            # Skip first two residues - backexchange
            if pos < start + 2:
                continue
                
            matrix[idx, pos] = 1
    
    # Create mapping
    peptide_map = {
        idx: {
            'sequence': peptides.iloc[idx]['Sequence'],
            'start': peptides.iloc[idx]['Start'],
            'end': peptides.iloc[idx]['End'],
            'n_exchangeable': int(matrix[idx].sum())  # Count exchangeable sites
        }
        for idx in range(len(peptides))
    }
    
    return matrix, peptide_map

def get_exchangeable_mask(protein_sequence):
    """
    Create mask for exchangeable residues
    
    Parameters:
    -----------
    protein_sequence : str
        Full protein sequence
        
    Returns:
    --------
    np.ndarray
        Boolean mask where True indicates exchangeable residue
    """
    mask = np.ones(len(protein_sequence), dtype=bool)
    
    # First two residues have fast back-exchange
    mask[0:2] = False
    
    # Prolines have no exchangeable amide
    for i, aa in enumerate(protein_sequence):
        if aa == 'P':
            mask[i] = False
            
    return mask

def load_hdx_data(filepath, include_fd=False):
    """
    Load HDX-MS data from CSV file, adding replicate numbers
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    include_fd : bool
        Whether to include fully deuterated controls
        
    Returns:
    --------
    pd.DataFrame
        Processed HDX data with replicates identified
    """
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Select key columns
    key_cols = ['State', 'DeutTime', 'Start', 'End', 'Sequence', 'Uptake', 'Experiment']
    df = df[key_cols].copy()
    
    # Convert timepoints to numeric, handling any time unit suffixes
    df['DeutTime'] = pd.to_numeric(df['DeutTime'].str.extract('(\d+)')[0], errors='coerce')
    
    # Identify FD controls (rows with NaN timepoints)
    fd_mask = df['DeutTime'].isna()
    
    if include_fd:
        # Store FD data separately
        fd_data = df[fd_mask].copy()
        # Could process FD data here if needed
    
    # Remove FD controls from main dataset
    df = df[~fd_mask]
    
    # Add replicate numbers within each state/time/peptide group
    df['Replicate'] = df.groupby(['State', 'DeutTime', 'Start', 'End'])['Experiment'].transform(
        lambda x: pd.factorize(x)[0] + 1
    )
    
    # Sort by peptide position, time, and replicate
    df = df.sort_values(['Start', 'End', 'DeutTime', 'Replicate'])
    
    if include_fd:
        return df, fd_data
    return df

def get_uptake_matrices(df):
    """
    Create matrices of uptake values including replicate information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed HDX data
        
    Returns:
    --------
    dict
        Contains:
        - mean: Mean uptake matrix (n_peptides, n_timepoints)
        - std: Standard deviation matrix (n_peptides, n_timepoints)
        - replicates: List of uptake matrices for each replicate
        - n_replicates: Number of replicates per condition
    list
        Timepoints
    """
    # Get unique peptides and timepoints
    peptides = df[['Start', 'End', 'Sequence']].drop_duplicates().reset_index(drop=True)
    timepoints = sorted(df['DeutTime'].unique())
    n_replicates = int(df['Replicate'].max())
    
    # Initialize matrices
    uptake_mean = np.zeros((len(peptides), len(timepoints)))
    uptake_std = np.zeros((len(peptides), len(timepoints)))
    uptake_replicates = []
    
    # Create a matrix for each replicate
    for rep in range(1, n_replicates + 1):
        rep_matrix = np.zeros((len(peptides), len(timepoints)))
        uptake_replicates.append(rep_matrix)
    
    # Fill matrices
    for pep_idx in range(len(peptides)):
        peptide = peptides.iloc[pep_idx]
        mask = (
            (df['Start'] == peptide['Start']) & 
            (df['End'] == peptide['End'])
        )
        
        for time_idx, time in enumerate(timepoints):
          time_mask = mask & (df['DeutTime'] == time)
          
          if time == 0:
              # For t=0, always set everything to 0 (undeuterated)
              uptake_mean[pep_idx, time_idx] = 0
              uptake_std[pep_idx, time_idx] = 0
              for rep_matrix in uptake_replicates:
                  rep_matrix[pep_idx, time_idx] = 0
                  
          else:
              # Get values for all replicates
              replicate_values = df[time_mask]['Uptake'].values
              
              # Set values
              if len(replicate_values) > 0:
                  # If we have measurements, use them
                  uptake_mean[pep_idx, time_idx] = np.mean(replicate_values)
                  uptake_std[pep_idx, time_idx] = np.std(replicate_values)
                  
                  # Fill replicate matrices
                  for rep_idx, rep in enumerate(range(1, n_replicates + 1)):
                      rep_mask = time_mask & (df['Replicate'] == rep)
                      if rep_mask.any():
                          uptake_replicates[rep_idx][pep_idx, time_idx] = df[rep_mask]['Uptake'].iloc[0]
    
    return {
        'mean': uptake_mean,
        'std': uptake_std,
        'replicates': uptake_replicates,
        'n_replicates': n_replicates
    }, timepoints

def process_hdx_dataset(filepath, fasta_filepath, protein_id=None):
    """
    Process complete HDX-MS dataset
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    fasta_filepath : str
        Path to FASTA file containing protein sequence
    protein_id : str, optional
        Protein ID in FASTA file
        
    Returns:
    --------
    dict
        Processed data including:
        - peptide_matrix
        - uptake_data (contains mean, std, and replicate matrices)
        - timepoints
        - peptide_map
        - protein_sequence
        - exchangeable_mask
        - raw_data
    """
    # Load HDX data
    df = load_hdx_data(filepath)
    
    # Load protein sequence
    sequences = read_fasta(fasta_filepath)
    if protein_id is None:
        protein_id = list(sequences.keys())[0]
    protein_sequence = sequences[protein_id]
    
    # Create exchangeable residue mask
    exchangeable_mask = get_exchangeable_mask(protein_sequence)
    
    # Create matrices
    peptide_matrix, peptide_map = create_peptide_matrix(
        df, 
        protein_sequence, 
        protein_length=len(protein_sequence)
    )
    
    # Get uptake data
    uptake_data, timepoints = get_uptake_matrices(df)
    
    return {
        'peptide_matrix': peptide_matrix,
        'uptake_data': uptake_data,
        'timepoints': timepoints,
        'peptide_map': peptide_map,
        'protein_sequence': protein_sequence,
        'exchangeable_mask': exchangeable_mask,
        'raw_data': df
    }

def prepare_replicate_data(hdx_dataset):
        """
        Convert replicate data from dictionary format to tensor format suitable for model
        
        Parameters:
        -----------
        hdx_dataset : dict
            Dictionary containing HDX data with replicates stored in hdx_dataset['uptake_data']['replicates']
        
        Returns:
        --------
        torch.Tensor
            Shape (n_replicates, n_peptides, n_timepoints)
        """
        replicates_dict = hdx_dataset['uptake_data']['replicates']
        n_replicates = len(replicates_dict)

        # Determine the dimensions of the replicate tensor
        n_peptides, n_timepoints = replicates_dict[0].shape  # Use replicates_dict[0]

        # Initialize the replicate tensor
        replicate_tensor = torch.zeros(n_replicates, n_peptides, n_timepoints)

        # Fill tensor with replicate data
        for rep_idx in range(n_replicates):  # Changed to iterate from 0 to n_replicates - 1
          replicate_tensor[rep_idx] = torch.tensor(replicates_dict[rep_idx])
      
        return replicate_tensor



