"""
Clustering algorithms for Mine Detection dataset
Unsupervised learning to discover patterns
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    calinski_harabasz_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


class ClusteringAnalysis:
    """
    Clustering analysis for mine detection dataset
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.labels = {}
        self.metrics = {}
        
    def perform_kmeans(self,
                      X: np.ndarray,
                      n_clusters: int = 5,
                      n_init: int = 10) -> KMeans:
        """
        Perform K-Means clustering
        
        Args:
            X: Feature array
            n_clusters: Number of clusters
            n_init: Number of initializations
            
        Returns:
            Fitted KMeans model
        """
        print(f"Performing K-Means with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=self.random_state
        )
        
        labels = kmeans.fit_predict(X)
        
        self.models['kmeans'] = kmeans
        self.labels['kmeans'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        
        self.metrics['kmeans'] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski,
            'inertia': kmeans.inertia_
        }
        
        print(f"✓ K-Means completed")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        
        return kmeans
    
    def find_optimal_k(self,
                      X: np.ndarray,
                      k_range: range = range(2, 11)) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Find optimal number of clusters using elbow method and silhouette
        
        Args:
            X: Feature array
            k_range: Range of k values to test
            
        Returns:
            Tuple of (figure, metrics dataframe)
        """
        print("Finding optimal K...")
        
        inertias = []
        silhouettes = []
        davies_bouldins = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            davies_bouldins.append(davies_bouldin_score(X, labels))
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'K': list(k_range),
            'Inertia': inertias,
            'Silhouette': silhouettes,
            'Davies-Bouldin': davies_bouldins
        })
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow plot
        axes[0].plot(k_range, inertias, marker='o', linewidth=2)
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(k_range, silhouettes, marker='s', linewidth=2, color='green')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin plot (lower is better)
        axes[2].plot(k_range, davies_bouldins, marker='^', linewidth=2, color='red')
        axes[2].set_xlabel('Number of Clusters (K)')
        axes[2].set_ylabel('Davies-Bouldin Index')
        axes[2].set_title('Davies-Bouldin Index (lower is better)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print("\nOptimal K Analysis:")
        print(results_df.to_string(index=False))
        
        return fig, results_df
    
    def perform_dbscan(self,
                      X: np.ndarray,
                      eps: float = 0.5,
                      min_samples: int = 5) -> DBSCAN:
        """
        Perform DBSCAN clustering
        
        Args:
            X: Feature array
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Fitted DBSCAN model
        """
        print(f"Performing DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        self.models['dbscan'] = dbscan
        self.labels['dbscan'] = labels
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"✓ DBSCAN completed")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        # Calculate metrics only for non-noise points
        if n_clusters > 1 and n_noise < len(labels):
            mask = labels != -1
            if np.sum(mask) > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
                
                self.metrics['dbscan'] = {
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise
                }
                
                print(f"  Silhouette Score: {silhouette:.4f}")
                print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return dbscan
    
    def perform_hierarchical(self,
                           X: np.ndarray,
                           n_clusters: int = 5,
                           linkage_method: str = 'ward') -> AgglomerativeClustering:
        """
        Perform Hierarchical clustering
        
        Args:
            X: Feature array
            n_clusters: Number of clusters
            linkage_method: Linkage method ('ward', 'complete', 'average')
            
        Returns:
            Fitted AgglomerativeClustering model
        """
        print(f"Performing Hierarchical Clustering ({linkage_method})...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        
        labels = hierarchical.fit_predict(X)
        
        self.models['hierarchical'] = hierarchical
        self.labels['hierarchical'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        self.metrics['hierarchical'] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
        
        print(f"✓ Hierarchical Clustering completed")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return hierarchical
    
    def plot_dendrogram(self, 
                       X: np.ndarray,
                       method: str = 'ward',
                       figsize: Tuple[int, int] = (12, 6)):
        """
        Plot dendrogram for hierarchical clustering
        
        Args:
            X: Feature array
            method: Linkage method
            figsize: Figure size
        """
        linkage_matrix = linkage(X, method=method)
        
        plt.figure(figsize=figsize)
        dendrogram(linkage_matrix)
        plt.title(f'Hierarchical Clustering Dendrogram ({method})')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_clusters_2d(self,
                             X: np.ndarray,
                             algorithm: str = 'kmeans',
                             figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize clusters in 2D using PCA
        
        Args:
            X: Feature array
            algorithm: Clustering algorithm to visualize
            figsize: Figure size
        """
        if algorithm not in self.labels:
            raise ValueError(f"Algorithm '{algorithm}' not found. Run clustering first.")
        
        labels = self.labels[algorithm]
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=labels, cmap='viridis', 
                            alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{algorithm.upper()} Clustering (2D PCA Projection)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def compare_with_true_labels(self,
                                y_true: np.ndarray,
                                algorithm: str = 'kmeans') -> float:
        """
        Compare clustering results with true labels using ARI
        
        Args:
            y_true: True labels
            algorithm: Clustering algorithm
            
        Returns:
            Adjusted Rand Index
        """
        if algorithm not in self.labels:
            raise ValueError(f"Algorithm '{algorithm}' not found")
        
        labels = self.labels[algorithm]
        ari = adjusted_rand_score(y_true, labels)
        
        print(f"\nAdjusted Rand Index ({algorithm}): {ari:.4f}")
        print("(1.0 = perfect match, 0.0 = random, negative = worse than random)")
        
        return ari
    
    def compare_all_algorithms(self) -> pd.DataFrame:
        """
        Compare all clustering algorithms
        
        Returns:
            Comparison dataframe
        """
        results = []
        
        for algo_name, metrics in self.metrics.items():
            result = {'Algorithm': algo_name.upper()}
            result.update(metrics)
            results.append(result)
        
        comparison_df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("CLUSTERING ALGORITHMS COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70)
        print("\nNote: Higher Silhouette Score is better")
        print("      Lower Davies-Bouldin Index is better")
        print("="*70 + "\n")
        
        return comparison_df
    
    def plot_cluster_distributions(self, 
                                  y_true: np.ndarray,
                                  algorithm: str = 'kmeans'):
        """
        Plot distribution of true labels within each cluster
        
        Args:
            y_true: True labels
            algorithm: Clustering algorithm
        """
        if algorithm not in self.labels:
            raise ValueError(f"Algorithm '{algorithm}' not found")
        
        labels = self.labels[algorithm]
        
        # Create confusion-like matrix
        unique_clusters = np.unique(labels[labels != -1])  # Exclude noise
        unique_classes = np.unique(y_true)
        
        matrix = np.zeros((len(unique_clusters), len(unique_classes)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = labels == cluster
            for j, cls in enumerate(unique_classes):
                matrix[i, j] = np.sum((y_true == cls) & mask)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'Class {c}' for c in unique_classes],
                   yticklabels=[f'Cluster {c}' for c in unique_clusters])
        plt.xlabel('True Mine Type')
        plt.ylabel('Cluster')
        plt.title(f'{algorithm.upper()}: Distribution of True Labels in Clusters')
        plt.tight_layout()
        
        return plt.gcf()


if __name__ == "__main__":
    # Example usage
    from data_loader import create_sample_data
    from preprocessing import MineDataPreprocessor
    
    print("Creating sample data...")
    df = create_sample_data(338)
    X = df[['V', 'H', 'S']]
    y = df['M']
    
    # Preprocess
    preprocessor = MineDataPreprocessor()
    X_scaled, y_encoded = preprocessor.prepare_for_neural_network(X, y)
    
    # Clustering analysis
    clustering = ClusteringAnalysis(random_state=42)
    
    # Find optimal K
    fig, results = clustering.find_optimal_k(X_scaled)
    plt.show()
    
    # Perform K-Means
    clustering.perform_kmeans(X_scaled, n_clusters=5)
    
    # Visualize
    fig = clustering.visualize_clusters_2d(X_scaled, 'kmeans')
    plt.show()
    
    # Compare with true labels
    ari = clustering.compare_with_true_labels(y_encoded, 'kmeans')