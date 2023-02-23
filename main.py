import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    print("Performing initial dimensionality reduction...")
    X = PCA(heart.X, 100)
    # Your code
    
    print("Fitting data to k-means model...")
    init_type = 'random' #Select centriod initialization method
    print('number of clusters: ', n_classifiers, ', initialization type: ', init_type)
    
    clusterings = []
    silhouettes = []
    
    #Running kmeans 10 times to select best fit
    for i in range(0,10):
        k_means = KMeans(n_classifiers,init=init_type)
        clusters = k_means.fit(X)
        clusterings.append(clusters)
        silhouettes.append(k_means.silhouette(clusters, X))

    
    #Select fit with best silhoette coeff
    max_silhouette = max(silhouettes)
    best_index = -1
    for i in range(0,len(silhouettes)):
        if max_silhouette == silhouettes[i]:
            best_index = int(i)
            break
        
    
    clusters = clusterings[best_index]
        
    
    print("Performing additional dimensionality reduction for plotting...")
    
    
    lengths = []
    for i in range(0, len(clusters)):
        lengths.append(len(clusters[i]))
        
    clustered_X = []
    
    for i in range(0,len(clusters)):
        for j in range(0, len(clusters[i])):
            clustered_X.append(clusters[i][j])
            
    clustered_X = PCA(clustered_X, 2)
    plot_x = clustered_X[:,0]
    plot_y = clustered_X[:,1]
    
    
        
    print("Plotting data...")
        
    visualize_cluster(plot_x, plot_y, lengths)
    
    print('Silhouette coefficient: ', max_silhouette)
        

def visualize_cluster(x, y, clustering):
    #Your code
    marker_combos = [['b', '.'],['g', '.'],['r', '.'],['b', 'v'],['g', 'v'],
                     ['r', 'v'],['b', 'x'],['g', 'x'],['r', 'x']]
    
    
    fig, ax = plt.subplots()
    clustering = [0] + clustering
    for i in range(0,len(clustering)-1):
        new_x = x[clustering[i]:clustering[i+1]+clustering[i]]
        new_y = y[clustering[i]:clustering[i+1]+clustering[i]]
        
        label = "Clustering " + str(i)
        ax.scatter(new_x, new_y, c=marker_combos[i][0],
                    marker=marker_combos[i][1], label = label)
        ax.set_xlim([-30 , 30])
        ax.set_ylim([-20 , 60])
        
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.show()
    

if __name__ == '__main__':
    main()
