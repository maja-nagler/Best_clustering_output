# Key difference to plot_clusters.py:
# 1. Single species instead of multiple species pipeline
# 2. Specific directory structure and paths for bengalese finch data
# 3. Simplified parameter handling without species loop


import matplotlib

matplotlib.use('Agg')  # Direct download of plot files instead of displaying in window
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

# CONFIGURATION - Change these values as needed
N_CLUSTERS = 6  # Number of clusters to visualise
N_EXAMPLES = 6  # Number of audio examples per cluster
USE_BEST_QUALITY = False  # True = best quality clusters, False = largest clusters


def calculate_cluster_quality(cluster_id, embeddings, clusters):
    # Simple quality score based on cluster variance
    cluster_mask = clusters == cluster_id
    cluster_embeddings = embeddings[cluster_mask]

    if len(cluster_embeddings) < 5:
        return 0.0

    # Checking the variance for clusters (lower variance = higher quality)
    cluster_center = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
    compactness = 1.0 / (np.std(distances) + 0.01)  # the higher the value the more compact the cluster

    # Possibility of adding a size filter
    size = len(cluster_embeddings)
    size_filtering = min(1.0, size / 100) * max(0.5, min(1.0, 500 / size))

    return compactness  # * size_filtering


def plot_bengalese_clusters():

    # Species loop from the original version removed
    specie = 'bengalese_finch1'
    base_path = f'/home/mayatoja/repertoire_embedder_new/paper_experiments/{specie}'
    print('Beginning to plot clusters')

    # Changed path to match the repository's structure
    csv_path = f'{base_path}/{specie}.csv'
    encodings_path = './encodings_bengalese_finch1_16_adapted.npy'
    audio_path = f'{base_path}/finch1/'

    # Loading existing embeddings and data
    dic = np.load(encodings_path, allow_pickle=True).item()
    idxs, encodings = dic['idxs'], dic['encodings']
    df = pd.read_csv(csv_path)

    if 'umap8' in dic:
        # CHecking if using the pre-computed 8-dimensional data is possible
        X = dic['umap8']
    else:
        # We need to create the 8-dimensional data ourselves
        print('Computing 8D UMAP projection for clustering...')
        # fit_transform applies the reduction to our encoding data
        X = umap.UMAP(n_components=8).fit_transform(encodings)  # reducing the data to 8 dimensions

    # Performing HDBSCAN clustering
    print('Performing HDBSCAN clustering...')
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        cluster_selection_epsilon=0.1,
        core_dist_n_jobs=-1,
        cluster_selection_method='leaf'  # removing 'eom' as not relevant
    ).fit_predict(X)

    df_filtered = df.loc[idxs].reset_index(drop=True)  # Keeping only rows with embeddings, renumber from 0
    df_filtered['cluster'] = clusters.astype(int)  # Adding cluster labels as a new column

    # Calculating cluster statistics
    cluster_counts = pd.Series(clusters).value_counts()
    valid_clusters = cluster_counts[cluster_counts.index != -1].index.tolist()

    print(f'Found {len(valid_clusters)} valid clusters')
    print('Cluster sizes:', cluster_counts.head(10))

    # Select clusters based on configuration
    if USE_BEST_QUALITY:
        print('Using best quality clusters...')
        # Calculate quality for all clusters
        cluster_quality = {}
        for cluster_id in valid_clusters:
            cluster_quality[cluster_id] = calculate_cluster_quality(cluster_id, encodings, clusters)

        # Sort by quality and take top N
        sorted_clusters = sorted(cluster_quality.items(), key=lambda x: x[1], reverse=True)
        n_to_plot = min(N_CLUSTERS, len(valid_clusters))
        top_clusters = [cluster_id for cluster_id, quality in sorted_clusters[:n_to_plot]]

        print('Top quality clusters:')
        for cluster_id, quality in sorted_clusters[:n_to_plot]:
            size = cluster_counts[cluster_id]
            print(f'  Cluster {cluster_id}: quality={quality:.2f}, size={size}')
    else:
        print('Using largest clusters...')
        # Take largest clusters by size
        n_to_plot = min(N_CLUSTERS, len(valid_clusters))
        top_clusters = cluster_counts.head(n_to_plot).index.tolist()
        if -1 in top_clusters:  # Remove noise cluster
            top_clusters.remove(-1)
            if len(valid_clusters) > n_to_plot:
                top_clusters.append(valid_clusters[n_to_plot])

    print(f'Plotting {len(top_clusters)} clusters: {top_clusters}')

    # Setting up frontend (tool to convert audio files into visual spectrograms for plotting)

    sr, nfft, sampleDur, nMel = 32000, 1024, 2.0, 128
    frontend = models.frontend['logMel'](sr, nfft, sampleDur, nMel)

    # Creating the output figure
    fig, axes = plt.subplots(nrows=len(top_clusters), ncols=1,
                             figsize=(12, 2 * len(top_clusters)), dpi=150)

    if len(top_clusters) == 1:
        axes = [axes]

    for i, cluster_id in enumerate(top_clusters):
        print(f'Processing cluster {cluster_id}...')

        # Examples from this cluster
        cluster_df = df_filtered[df_filtered.cluster == cluster_id]

        if len(cluster_df) == 0:
            continue

        # Use configured number of examples
        n_samples = min(N_EXAMPLES, len(cluster_df))
        sampled_df = cluster_df.sample(n_samples) if len(cluster_df) > n_samples else cluster_df

        # Try/except blocks for error handling
        try:
            dataset = u.Dataset(sampled_df, audio_path, sr, sampleDur)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=u.collate_fn)

            for j, (x, name) in enumerate(loader):
                if j >= N_EXAMPLES:
                    break

                try:
                    # Computting spectrogram
                    with torch.no_grad():
                        spec = frontend(x).squeeze().numpy()

                    # Plotting
                    extent = [j, j + 1, 0, 1]
                    axes[i].imshow(spec, extent=extent, origin='lower', aspect='auto',
                                   cmap='viridis', vmin=np.quantile(spec, 0.1),
                                   vmax=np.quantile(spec, 0.9))

                except Exception as e:
                    print(f'Error processing example {j} from cluster {cluster_id}: {e}')
                    continue

            # Formatting axis
            axes[i].set_xlim(0, N_EXAMPLES)
            axes[i].set_ylim(0, 1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

            # Vertical lines between examples
            for k in range(1, N_EXAMPLES):
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
            axes[i].set_xlim(0, N_EXAMPLES)
            axes[i].set_ylim(0, 1)

    plt.tight_layout()

    # Save with descriptive filename based on settings
    method = 'quality' if USE_BEST_QUALITY else 'size'
    output_path = f'{base_path}/clusters_spectrograms_new_{N_CLUSTERS}_{method}_{N_EXAMPLES}ex.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Cluster spectrograms saved to: {output_path}')

    # summary plot showing cluster distribution
    create_cluster_summary(df_filtered, base_path, method)


def create_cluster_summary(df_filtered, base_path, method=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Cluster size distribution
    cluster_counts = df_filtered['cluster'].value_counts().sort_index()
    valid_clusters = cluster_counts[cluster_counts.index != -1]

    ax1.bar(range(len(valid_clusters)), valid_clusters.values)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of vocalizations')
    ax1.set_title('Cluster Size Distribution')
    ax1.set_xticks(range(0, len(valid_clusters), max(1, len(valid_clusters) // 10)))

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
                             color='white' if value > plot_data.values.max() / 2 else 'black',
                             fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No ground truth labels available',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Ground Truth Labels')

    plt.tight_layout()

    # Save with descriptive filename
    if method:
        summary_path = f'{base_path}/cluster_summary_new_{N_CLUSTERS}_{method}.png'
    else:
        summary_path = f'{base_path}/cluster_summary_new.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Cluster summary saved to: {summary_path}')


plot_bengalese_clusters()
