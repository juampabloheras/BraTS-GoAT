import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import pandas as pd
import pickle


# Function to assign dummy clusters to data for debugging
def assign_dummy_clusters(in_dir, out_dir, num_clusters, overwrite = False):
    
    # Load data, shuffle data list, and split clusters
    file_names = os.listdir(in_dir)
    random.shuffle(file_names)
    clusters = np.array_split(file_names, num_clusters)
    cluster_lengths = [len(cluster) for cluster in clusters]

    cluster_mapping = {}
    for i, cluster_list in enumerate(clusters):
        cluster_mapping[i] = cluster_list


    if os.path.exists(save_path) and not overwrite:
        print(f"File {save_path} already exists. Use overwrite=True to overwrite the file.")
        return

    with open(save_path, 'wb') as file:
        pickle.dump(cluster_mapping, file)

    print(f"Clusters have been saved to {save_path}")
    print(f"Cluster sizes: {cluster_lengths}")


if __name__ == '__main__':
    
    in_dir = '/gscratch/scrubbed/juampablo/BraTS-GoAT/DATA/training'
    save_path = '/gscratch/kurtlab/juampablo/BraTS-GoAT/neuralnet/cluster_dict.pkl'
    num_clusters = 3

    assign_dummy_clusters(in_dir, save_path, num_clusters, overwrite=True)