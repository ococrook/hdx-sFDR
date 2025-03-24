import statistical_inference
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_alpha_effects(peptides, coords, plddt, alphas=np.linspace(0, 1, 21)):
    baseline_qvalues = statistical_inference.compute_qvalues(peptides[:,2])
    
    results = []
    for alpha in alphas:
        # Get component weights separately to analyze their contribution
        weights, seq_weights, struct_weights, conf_weights = statistical_inference.calculate_weights(
            peptides, coords, plddt,
            lambda_seq=1,
            lambda_struct=2,
            alpha=alpha
        )
        
        # Calculate weighted p-values
        weighted_pvalues, qvalues = statistical_inference.calculate_weighted_pvalues(
            peptides, weights, transform_sum=True
        )
        
        # Store component contributions for analysis
        results.append({
            'alpha': alpha,
            'seq_contribution': np.mean(alpha * seq_weights),
            'struct_contribution': np.mean((1 - alpha) * struct_weights * np.mean(conf_weights)),
            'original_p': peptides[:,2],
            'original_q': statistical_inference.compute_qvalues(peptides[:,2]),
            'weighted_pvalues': weighted_pvalues,
            'qvalues': qvalues,
        })
    
    return pd.DataFrame(results)

def plot_weight_comparison(results_df):
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Weighting Effects Across Alpha Values', fontsize=14)
    
    # Plot 1: Component contributions
    ax1 = axes[0]
    ax1.plot(results_df['alpha'], results_df['seq_contribution'], label='Sequence')
    ax1.plot(results_df['alpha'], results_df['struct_contribution'], label='Structure')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Mean Weight Contribution')
    ax1.legend()
    ax1.set_title('Weight Component Contributions')
    
    # Plot 2: Compare significant hits at threshold
    ax2 = axes[1]
    sig_counts = []
    for idx, row in results_df.iterrows():
        orig_sig = np.sum(row['original_q'] < 0.01)
        weighted_sig = np.sum(row['qvalues'] < 0.01)
        sig_counts.append({
            'alpha': row['alpha'],
            'original': orig_sig,
            'weighted': weighted_sig
        })
    sig_df = pd.DataFrame(sig_counts)
    
    ax2.plot(sig_df['alpha'], sig_df['original'], label='Original', linestyle='--')
    ax2.plot(sig_df['alpha'], sig_df['weighted'], label='Weighted')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Number of Significant Hits (p < 0.01)')
    ax2.legend()
    ax2.set_title('Significant Hits Comparison')

    
    plt.tight_layout()
    return fig

def evaluate_multiple_regions(hdx_dataset, regions, coords, plddt, alphas=np.linspace(0, 1, 11)):
    all_results = {}
    
    for start, end in regions:
        # Prepare peptide map for this region
        peptide_map = statistical_inference.prepare_peptide_map(
            hdx_dataset['raw_data'], 
            effect_positions=[(start, end)]
        )
        
        peptides = statistical_inference.process_peptide_map(peptide_map)

        # Evaluate alphas for this region
        region_results = []
        for alpha in alphas:
            weights, seq_weights, struct_weights, conf_weights = statistical_inference.calculate_weights(
                peptides, coords, plddt,
                lambda_seq=1,
                lambda_struct=2,
                alpha=alpha
            )
            
            weighted_pvalues, qvalues = statistical_inference.calculate_weighted_pvalues(
                peptides, weights, transform_sum=True
            )
            
            region_results.append({
                'alpha': alpha,
                'region': f"{start}-{end}",
                'seq_contribution': np.mean(alpha * seq_weights),
                'struct_contribution': np.mean((1 - alpha) * struct_weights),
                'conf_effect': np.mean(conf_weights),
                'original_p': peptides[:,2],
                'original_q': statistical_inference.compute_qvalues(peptides[:,2]),
                'weighted_pvalues': weighted_pvalues,
                'qvalues': qvalues,
            })
            
        all_results[f"{start}-{end}"] = region_results
    
    return all_results

def plot_regional_comparison(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Weighting Effects Across Regions', fontsize=14)
    
    # Convert results to a more convenient format
    plot_data = []
    for region, results in all_results.items():
        for result in results:
            plot_data.append({
                'alpha': result['alpha'],
                'region': region,
                'seq_contribution': result['seq_contribution'],
                'struct_contribution': result['struct_contribution'],
                'sig_hits': np.sum(result['weighted_pvalues'] < 0.01),
                'mean_pvalue': np.mean(result['weighted_pvalues'])
            })
    plot_df = pd.DataFrame(plot_data)
    
    # Plot 1: Sequence contribution across regions
    sns.lineplot(data=plot_df, x='alpha', y='seq_contribution', 
                hue='region', ax=axes[0,0])
    axes[0,0].set_title('Sequence Contribution by Region')
    
    # Plot 2: Structure contribution across regions
    sns.lineplot(data=plot_df, x='alpha', y='struct_contribution', 
                hue='region', ax=axes[0,1])
    axes[0,1].set_title('Structure Contribution by Region')
    
    # Plot 3: Number of significant hits
    sns.lineplot(data=plot_df, x='alpha', y='sig_hits', 
                hue='region', ax=axes[1,0])
    axes[1,0].set_title('Significant Hits (p < 0.01)')
    
    # Plot 4: Mean p-value
    sns.lineplot(data=plot_df, x='alpha', y='mean_pvalue', 
                hue='region', ax=axes[1,1])
    axes[1,1].set_title('Mean P-value')
    
    plt.tight_layout()
    return fig
  
def prepare_for_r_export(all_results):
    # Create flattened data structure
    flat_data = []
    
    for region, results in all_results.items():
        for result in results:
            # Base metrics
            base_row = {
                'alpha': result['alpha'],
                'region': region,
                'seq_contribution': result['seq_contribution'],
                'struct_contribution': result['struct_contribution'],
                'conf_effect': result['conf_effect']
            }
            
            # Expand p-values and q-values
            for i, (orig_p, orig_q, weighted_p, weighted_q) in enumerate(zip(
                result['original_p'],
                result['original_q'],
                result['weighted_pvalues'],
                result['qvalues']
            )):
                row = base_row.copy()
                row.update({
                    'peptide_index': i,
                    'original_p': orig_p,
                    'original_q': orig_q,
                    'weighted_p': weighted_p,
                    'weighted_q': weighted_q
                })
                flat_data.append(row)
    
    df = pd.DataFrame(flat_data)
    return df
