#Key modifications:
#1. Data structure and paths adapted to the directory
            #e.g. 'finch1/' subdirectory for audio files
#2. Fixed parameters added instead of command-line arguments for simplicity
            # Fixed bottleneck=256, nMel=128 based on available model weights


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import models, utils as u
import pandas as pd, numpy as np, torch
import argparse, os
from tqdm import tqdm
from sklearn import metrics
import umap, hdbscan
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# Fixed parameters
specie = 'bengalese_finch1'
bottleneck = 256
nMel = 128
prcptl = 1
encoder = 'sparrow_encoder'
ncomp = 2  # 2D for visualization
frontend = 'logMel'


# Changed directory structure- original format: f'{args.specie}/{args.specie}.csv'
base_path = f'./repertoire_embedder_new/paper_experiments/{specie}'
csv_path = f'{base_path}/{specie}.csv'
audio_path = f'{base_path}/finch1/' 
encodings_path = f'{base_path}/encodings/'

# Loading data
df = pd.read_csv(csv_path)
print(len(df), 'available vocalizations')

## Checking if encodings already exist
modelname = f'{specie}_{bottleneck}_{frontend}{nMel}_{encoder}_decod2_BN_nomaxPool.stdc'
encodings_file = f'{encodings_path}encodings_{modelname[:-5]}.npy'

X = None  #suggests we don't have visualisation coordinates yet, X variable will store the UMAP projections for plotting

# Checking for pre-computed embeddings (to possibly save processing time)
if os.path.isfile(encodings_file):
    print('Loading existing encodings from', encodings_file)
    dic = np.load(encodings_file, allow_pickle=True).item()  # Loading the saved data
    idxs, encodings = dic['idxs'], dic['encodings']  # Extracting embeddings and their indices
    
    # Check if we also have pre-computed UMAP projection (saves even more time)
    if f'umap{ncomp}' in dic.keys():
        print('pre-computed UMAP exists')
        X = dic[f'umap{ncomp}']  # Using the saved UMAP data instead of computing it
    print('Loaded embeddings:', encodings.shape)
    print('Index mapping:', len(idxs), 'entries')
else:
    # Printing names of available encodings
    print('Encodings file not found:', encodings_file)
    print('Available encoding files:')
    if os.path.exists(encodings_path):
        for f in os.listdir(encodings_path):
            print('  ' + f)
    
    # Looking for alternative encodings based on the bottleneck  (here-256)
    alt_file = f'{encodings_path}encodings_{specie}_256_logMel128_{encoder}_decod2_BN_nomaxPool.npy'
    if os.path.isfile(alt_file):
        print('Using alternative encodings:', alt_file)
        dic = np.load(alt_file, allow_pickle=True).item()  # Loading alternative file
        idxs, encodings = dic['idxs'], dic['encodings']  # Extracting data from alternative file
        print('Loaded embeddings:', encodings.shape)
    else:
        print('Switching to generation of encodings with the following section')
        sys.exit(1)  # Stopping the section of the script

## Creating UMAP projection if existing one not available 
if X is None:
    print('Computing UMAP projection to', str(ncomp) + 'D...')
    X = umap.UMAP(n_jobs=-1, n_components=ncomp, random_state=42).fit_transform(encodings)
    print('UMAP projection completed:', X.shape)
    
    # Saving updated encodings with UMAP
    dic[f'umap{ncomp}'] = X
    np.save(encodings_file.replace('.npy', f'_with_umap{ncomp}.npy'), dic)

## Performing HDBSCAN clustering
print('Performing HDBSCAN clustering...')
clusters = hdbscan.HDBSCAN(
    min_cluster_size=10, 
    min_samples=3, 
    cluster_selection_epsilon=0.1, 
    cluster_selection_method='leaf', 
    core_dist_n_jobs=-1
).fit_predict(X)

# Filtering the dataframe to match embeddings
df_filtered = df.loc[idxs].reset_index(drop=True) # Keeping only rows with embeddings, renumber from 0
df_filtered['cluster'] = clusters.astype(int)  # Adding cluster labels as a new column

# Analysis
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1) 
print('Found', n_clusters, 'clusters with', n_noise, 'noise points')

# Create a filter to find vocalisations that have known labels 
mask = ~df_filtered.label.isna()  # True for labeled vocalisations, False for unlabeled ones

# Cluster assignments for only the labeled vocalisations
labeled_clusters = clusters[mask]  # Model-assigned clusters for labeled data

# Getting the correct labels for the labeled vocalisations  
labeled_true = df_filtered.loc[mask, 'label']  # Human-annotated ground truth labels

if len(labeled_true) > 0:
    print('Computing clustering Performance Metrics:')
    
    #NMI: How much information the clusters share with true labels (0=random, 1=perfect match)
    nmi = metrics.normalized_mutual_info_score(labeled_true, labeled_clusters)
    
    #ARI: Similarity of cluster assignments with the true labels, accounting for chance (0=random, 1=perfect)
    ari = metrics.adjusted_rand_score(labeled_true, labeled_clusters)
    
    #Silhouette: How well-separated the clusters are (-1=bad, +1=excellent)
    silhouette = metrics.silhouette_score(encodings[mask], labeled_clusters)
    
    #Homogeneity: does each cluster contain only one type of vocalisation (0=mixed, 1=pure clusters)
    homogeneity = metrics.homogeneity_score(labeled_true, labeled_clusters)
    
    #Completeness: Whether all vocalisations of each type end up in the same cluster (0=scattered, 1=grouped)
    completeness = metrics.completeness_score(labeled_true, labeled_clusters)
    
    #V-measure: Harmonic mean of homogeneity and completeness
    v_measure = metrics.v_measure_score(labeled_true, labeled_clusters)
    
    print('Normalized Mutual Information:', round(nmi, 3))
    print('Adjusted Rand Index:', round(ari, 3))
    print('Silhouette Score:', round(silhouette, 3))
    print('Homogeneity:', round(homogeneity, 3))
    print('Completeness:', round(completeness, 3))
    print('V-Measure:', round(v_measure, 3))
    
    # Save results to tests.csv
    results_file = f'{base_path}/tests_bengalese.csv'
    with open(results_file, 'a') as f:
        frontend_name = str(bottleneck) + '_' + frontend + str(nMel)
        f.write(f'{specie},{frontend_name},{nmi},{10},{3},{0.1},leaf,{ncomp}\n')
    print('Results saved to', results_file)
 
    print('Creating Visualizations:')
    
    # Creating directory
    proj_dir = f'{base_path}/projections'
    os.makedirs(proj_dir, exist_ok=True)
    
    # Plot 1: Clusters
    plt.figure(figsize=(20, 10))
    
    # Plotting noise points
    noise_mask = clusters == -1
    if np.any(noise_mask):
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1], s=2, alpha=0.2, color='Grey', label='Noise')
    
    # Plotting clusters
    valid_cluster_ids = sorted([c for c in set(clusters) if c != -1])
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(valid_cluster_ids)))
    for i, cluster_id in enumerate(valid_cluster_ids):
        mask = clusters == cluster_id
        plt.scatter(X[mask, 0], X[mask, 1], s=2, c=[cluster_colors[i]], alpha=0.7, 
                   label='Cluster ' + str(cluster_id) + ' (' + str(np.sum(mask)) + ')')
    
    plt.title(specie + ' - HDBSCAN Clusters')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    cluster_plot = f'{proj_dir}/{modelname[:-5]}_projection_clusters.png'
    plt.savefig(cluster_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print('Cluster plot saved:', cluster_plot)
    
    # Plot 2: Ground truth labels
    plt.figure(figsize=(20, 10))
    
    # Plotting unlabeled points
    unlabeled_mask = df_filtered.label.isna()
    if np.any(unlabeled_mask):
        plt.scatter(X[unlabeled_mask, 0], X[unlabeled_mask, 1], s=2, alpha=0.2, color='Grey', label='Unlabeled')
    
    # Plotting labeled points by ground truth
    unique_labels = sorted([x for x in df_filtered.label.unique() if pd.notna(x)])
    label_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = df_filtered.label == label
        if np.any(mask):
            plt.scatter(X[mask, 0], X[mask, 1], s=4, c=[label_colors[i]], alpha=0.7,
                       label=str(label) + ' (' + str(np.sum(mask)) + ')')
    
    plt.title(specie + ' - Ground Truth Labels')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    labels_plot = f'{proj_dir}/{modelname[:-5]}_projection_labels.png'
    plt.savefig(labels_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print('Labels plot saved:', labels_plot)
    
    # Analysis of cluster quality
    print('Cluster Quality Analysis')
    labelled = df_filtered[~df_filtered.label.isna()]
    goodClusters = []
    
    for label, grp in labelled.groupby('label'):
        if len(grp) == 0:
            continue
        cluster_counts = grp.groupby('cluster').size()
        total_counts = labelled.groupby('cluster').size()
        
        # Precision for each cluster containing this label
        precisions = cluster_counts / total_counts
        precisions = precisions.dropna()
        
        if len(precisions) > 0:
            best_cluster = precisions.idxmax()
            best_precision = precisions.max()
            
            # Number of clusters have with precision above 90%
            good_clusters_for_label = precisions[precisions > 0.9].index.tolist()
            goodClusters.extend(good_clusters_for_label)
            
            # Recall
            total_label_points = len(grp)
            points_in_best_cluster = len(grp[grp.cluster == best_cluster])
            recall = points_in_best_cluster / total_label_points
            
            print('Best precision for', label + ': cluster', best_cluster, 'with', (df_filtered.cluster==best_cluster).sum(), 'points')
            print('  Precision:', round(best_precision, 3), ', Recall:', round(recall, 3))

    goodClusters = list(set(goodClusters))  # Using set() to remove duplicates
    sorted_samples = df_filtered.cluster.isin(goodClusters).sum()
    total_samples = len(df_filtered)
    

    print(len(goodClusters), 'high-quality clusters would sort', str(sorted_samples) + '/' + str(total_samples), 'samples (' + str(round(100*sorted_samples/total_samples)) + '%)')
    if len(unique_labels) > 0:
        print(round(len(goodClusters)/len(unique_labels), 1), 'clusters per label on average')
    
    # Results path added
    results = {
        'umap_embedding': X,
        'cluster_labels': clusters,
        'df_with_clusters': df_filtered,
        'metrics': {
            'nmi': nmi if 'nmi' in locals() else None,
            'ari': ari if 'ari' in locals() else None,
            'silhouette': silhouette if 'silhouette' in locals() else None
        }
    }
    
    results_path = f'{base_path}/test_AE_results.npy'
np.save(results_path, results, allow_pickle=True)
print('Final results saved to:', results_path)
