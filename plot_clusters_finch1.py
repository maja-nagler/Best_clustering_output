
## Key difference to plot_clusters.py:
# 1. Single species instead of multiple species pipeline
# 2. Specific directory structure and paths for bengalese finch data
# 3. Simplified parameter handling without species loop


import matplotlib
matplotlib.use('Agg')  # Allows for direct download of plot files instead of displaying in window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import hdbscan
import umap
import models
import utils as u
import os
import soundfile as sf

def plot_bengalese_clusters():
    
    ## Plotting spectrograms of example vocalizations from different clusters
    
    # Species loop from the original version removed
    specie = 'bengalese_finch1'
    base_path = f'/repertoire_embedder_new/paper_experiments/{specie}'
    print('Beginning to plot clusters')    

    # Changed path to match the repository's structure
    csv_path = f'{base_path}/{specie}.csv'
    encodings_path = f'{base_path}/encodings/encodings_{specie}_256_logMel128_sparrow_encoder_decod2_BN_nomaxPool.npy'
    audio_path = f'{base_path}/finch1/'
    
    # Loading existing embeddings and data
    dic = np.load(encodings_path, allow_pickle=True).item()
    idxs, encodings = dic['idxs'], dic['encodings']
    df = pd.read_csv(csv_path)
    
    #UMAP
    if 'umap8' in dic:
        # CHecking if using the pre-computed 8-dimensional data is possible
        X = dic['umap8']
    else:
        # We need to create the 8-dimensional data ourselves
        print('Computing 8D UMAP projection for clustering...')
        # fit_transform applies the reduction to our encoding data
        X = umap.UMAP(n_components=8).fit_transform(encodings) #reducing the data to 8 dimensions
    
    # Performing HDBSCAN clustering 
    print('Performing HDBSCAN clustering...')
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=10, 
        min_samples=3, 
        cluster_selection_epsilon=0.1, 
        core_dist_n_jobs=-1, 
        cluster_selection_method='leaf'  #  removing 'eom' as not relevant
    ).fit_predict(X)
    

    df_filtered = df.loc[idxs].reset_index(drop=True)  # Keeping only rows with embeddings, renumber from 0
    df_filtered['cluster'] = clusters.astype(int)  # Adding cluster labels as a new column
    
    # Calculating cluster statistics
    cluster_counts = pd.Series(clusters).value_counts()
    valid_clusters = cluster_counts[cluster_counts.index != -1].index.tolist()
    
    print('Found', {len(valid_clusters)}, 'valid clusters')
    print('Cluster sizes:', cluster_counts.head(10))
    
    #Increasing the minimum number of clusters (from fixed 4 random clusters)
    n_clusters_to_plot = min(6, len(valid_clusters))
    top_clusters = cluster_counts.head(n_clusters_to_plot).index.tolist()
    if -1 in top_clusters:  # noise cluster removed
        top_clusters.remove(-1)
        if len(valid_clusters) > n_clusters_to_plot:
            top_clusters.append(valid_clusters[n_clusters_to_plot])

    print('Plotting top', {len(top_clusters)}, 'clusters:', {top_clusters})

    # Setting up frontend (tool to convert audio files into visual spectrograms for plotting)

    sr, nfft, sampleDur, nMel = 32000, 1024, 2.0, 128
    frontend = models.frontend['logMel'](sr, nfft, sampleDur, nMel)
    
    # Creating the output figure
    n_examples = 8  # Number of examples per cluster
    fig, axes = plt.subplots(nrows=len(top_clusters), ncols=1, 
                           figsize=(12, 2*len(top_clusters)), dpi=150)
    
    if len(top_clusters) == 1:
        axes = [axes]
    
    for i, cluster_id in enumerate(top_clusters):
        print(f'Processing cluster {cluster_id}...')
        
        # Examples from this cluster
        cluster_df = df_filtered[df_filtered.cluster == cluster_id]
        
        if len(cluster_df) == 0:
            continue
            
        # Defining sample number 
        n_samples = min(n_examples, len(cluster_df))
        sampled_df = cluster_df.sample(n_samples) if len(cluster_df) > n_samples else cluster_df

        # Try/except blocks for error handling
        try:
            dataset = u.Dataset(sampled_df, audio_path, sr, sampleDur)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=u.collate_fn)
            
          
            for j, (x, name) in enumerate(loader):
                if j >= n_examples:
                    break
                    
                try:
                    # Computting spectrogram
                    with torch.no_grad():
                        spec = frontend(x).squeeze().numpy()
                    
                    # Plotting 
                    extent = [j, j+1, 0, 1]
                    axes[i].imshow(spec, extent=extent, origin='lower', aspect='auto', 
                                  cmap='viridis', vmin=np.quantile(spec, 0.1), 
                                  vmax=np.quantile(spec, 0.9))
                                  
                except Exception as e:
                    print(f'Error processing example {j} from cluster {cluster_id}: {e}')
                    continue
            
            # Formatting axis
            axes[i].set_xlim(0, n_examples)
            axes[i].set_ylim(0, 1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # Vertical lines between examples
            for k in range(1, n_examples):
                axes[i].axvline(k, color='white', linewidth=1)
            
            # Labelling
            cluster_size = len(cluster_df)
            dominant_label = 'Mixed'
            if 'label' in cluster_df.columns:
                label_counts = cluster_df['label'].value_counts()
                if len(label_counts) > 0:
                    dominant_label = str(label_counts.index[0])
            
            axes[i].set_ylabel(f'Cluster {cluster_id}\n({cluster_size} vocs)\n{dominant_label}', 
                              fontsize=10)
            
        except Exception as e:
            print(f'Error processing cluster {cluster_id}: {e}')
            axes[i].text(0.5, 0.5, f'Error loading\nCluster {cluster_id}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xlim(0, n_examples)
            axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Saving the plot
    output_path = f'{base_path}/clusters_spectrograms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('Cluster spectrograms saved to path:', {output_path})

    # summary plot showing cluster distribution
    create_cluster_summary(df_filtered, base_path)

def create_cluster_summary(df_filtered, base_path):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cluster size distribution
    cluster_counts = df_filtered['cluster'].value_counts().sort_index()
    valid_clusters = cluster_counts[cluster_counts.index != -1]
    
    ax1.bar(range(len(valid_clusters)), valid_clusters.values)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of vocalizations')
    ax1.set_title('Cluster Size Distribution')
    ax1.set_xticks(range(0, len(valid_clusters), max(1, len(valid_clusters)//10)))
    
    # Plot 2: Ground truth label distribution within clusters 
    if 'label' in df_filtered.columns:
        # Create a confusion matrix style plot
        cluster_label_counts = pd.crosstab(df_filtered['cluster'], df_filtered['label'])
        
        # Select top clusters and labels for visualisation
        top_clusters = cluster_counts.head(10).index.tolist()
        if -1 in top_clusters:
            top_clusters.remove(-1)
        
        top_labels = df_filtered['label'].value_counts().head(10).index.tolist()
        
        # Create a table showing which song types appear most in each cluster
        plot_data = cluster_label_counts.loc[top_clusters, top_labels].fillna(0)
        
        im = ax2.imshow(plot_data.values, cmap='Blues', aspect='auto')
        
        # Labels
        ax2.set_xticks(range(len(plot_data.columns)))
        ax2.set_xticklabels([str(x)[:10] for x in plot_data.columns], rotation=45, ha='right')
        ax2.set_yticks(range(len(plot_data.index)))
        ax2.set_yticklabels([f'C{x}' for x in plot_data.index])
        ax2.set_xlabel('Ground Truth Labels')
        ax2.set_ylabel('Cluster ID')
        ax2.set_title('Cluster vs Ground Truth Labels')
        
        # colorbar
        plt.colorbar(im, ax=ax2, label='Count')
        
        # Text annotations for non-zero values
        for i in range(len(plot_data.index)):
            for j in range(len(plot_data.columns)):
                value = plot_data.iloc[i, j]
                if value > 0:
                    ax2.text(j, i, f'{int(value)}', ha='center', va='center',
                           color='white' if value > plot_data.values.max()/2 else 'black',
                           fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No ground truth labels available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Ground Truth Labels')
    
    plt.tight_layout()
    
    # Changed output format from 'pdf' to 'png'
    summary_path = f'{base_path}/cluster_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('Cluster summary saved to path:', {summary_path})

plot_bengalese_clusters()
