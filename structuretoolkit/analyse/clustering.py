import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from ase.io import read
from ase.atoms import Atoms
from ase.visualize import view
import matplotlib.pyplot as plt

def compute_cluster_labels(structure, num_clusters):
    """
    Compute hierarchical clustering labels for an ASE Atoms structure.
    
    Use case: Identification of inherently different parts of a single structure, i.e. separate slabs, specific phases, etc.
    Atomic distances are the sole defining metric used for clustering.

    Parameters:
        structure (Atoms): ASE Atoms object.
        num_clusters (int): Number of clusters to form.

    Returns:
        np.ndarray: Cluster labels for each atom.
    """
    if isinstance(structure, Atoms):
        # If structure is an ASE Atoms object, use it directly
        struct = structure
    else:
        raise ValueError("Invalid input for structure. Please provide an ASE Atoms object.")

    # Calculate the distance matrix
    distance_matrix = struct.get_all_distances(mic=True)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Get cluster labels for the specified number of clusters
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    return cluster_labels

def ase_view_clusters(structure, n_clusters, target_cluster_label=1):
    """
    Visualize a specific cluster in an ASE Atoms structure.

    Parameters:
        structure (Atoms): ASE Atoms object.
        target_cluster_label (int): Target cluster label to visualize (NOTE: STARTS AT 1, NOT 0 (scipy))
    """
    cluster_labels = compute_cluster_labels(structure, n_clusters)
    # Print or visualize the indices of the specified cluster label
    indices_of_cluster = np.where(cluster_labels == target_cluster_label)[0]
    # Visualize the cluster using ASE's view function
    view(structure[indices_of_cluster])
    
def plot_clusters(structure, n_clusters=1, projection=[1, 2], figsize=(30, 10)):
    """
    Plot clusters in a 2D scatter plot based on hierarchical clustering.

    Parameters:
        structure (Atoms): ASE Atoms object.
        n_clusters (int): Number of clusters to form.
        projection (list): List of two integers specifying axes for the scatter plot.
        figsize (tuple): Figure size.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute cluster labels
    cluster_labels = compute_cluster_labels(structure, n_clusters)

    for cluster_label in np.unique(cluster_labels):
        cluster_data = structure.positions[cluster_labels == cluster_label]
        ax.scatter(cluster_data[:, projection[0]], cluster_data[:, projection[1]], label=f'Cluster {cluster_label}')

    ax.set_xlabel(f'Axis {projection[0]}')
    ax.set_ylabel(f'Axis {projection[1]}')
    ax.set_title(f'2D Projection (Axis {projection[0]}-{projection[1]}) \nHierarchical Clusters ')
    ax.legend(loc=[1.05,0.9])

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Set axis limits to be tight
    ax.autoscale()

def get_structure_clusters(structure, n_clusters=2):
    """
    Split an ASE Atoms structure into multiple structures based on hierarchical clustering.

    Parameters:
        structure (Atoms): ASE Atoms object.
        n_clusters (int): Number of clusters to form.

    Returns:
        list: List of ASE Atoms structures, each corresponding to a cluster.
    """
    # Returns a list of structures 
    cluster_labels = compute_cluster_labels(structure, n_clusters)
    struct_list = []
    for target_cluster_label in np.unique(cluster_labels):
        target_cluster_label = 1
        indices_of_cluster = np.where(cluster_labels == target_cluster_label)[0]
        struct_list.append(structure.copy()[indices_of_cluster])
    return struct_list
