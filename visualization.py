import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from model import EmbeddingModel, DeepResidualEmbeddingModel
import plotly.graph_objects as go
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset
import umap
import umap.plot
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
from app import load_train_test_dataset
import os
import time

def visualize_umap(encoded_data, labels, patient_ids, results_dir):
    print("Computing UMAP embedding...")
    mapper = umap.UMAP(random_state=42, metric='cosine', n_neighbors=15, min_dist=0.1).fit(encoded_data)
    print("UMAP computation completed!")
    
    print("Creating UMAP plots...")
    umap.plot.points(mapper, labels=patient_ids)
    plt.savefig(os.path.join(results_dir, 'umap_plot_patient_ids.png'))
    plt.close()
    
    umap.plot.points(mapper, labels=labels)
    plt.savefig(os.path.join(results_dir, 'umap_plot_labels.png'))
    plt.close()
    print("UMAP plots saved!")

def visualize_embeddings(encoded_data, labels, results_dir):
    # Apply PCA to reduce dimensionality of data from embedding_dim -> 3d to make it easier to visualize!
    pca = PCA(n_components=3)
    encoded_data_3d = pca.fit_transform(encoded_data)

    scatter = go.Scatter3d(
        x=encoded_data_3d[:, 0],
        y=encoded_data_3d[:, 1],
        z=encoded_data_3d[:, 2],
        mode='markers',
        marker=dict(size=4, color=labels, colorscale='Viridis', opacity=0.8),
        text=labels, 
        hoverinfo='text',
    )

    # Create layout
    layout = go.Layout(
        title="TUH Dataset - Encoded and PCA Reduced 3D Scatter Plot",
        scene=dict(
            xaxis=dict(title="PC1"),
            yaxis=dict(title="PC2"),
            zaxis=dict(title="PC3"),
        ),
        width=1000, 
        height=750,
    )

    # Create figure and add scatter plot
    fig = go.Figure(data=[scatter], layout=layout)

    # Save the plot as HTML file instead of showing it
    fig.write_html(os.path.join(results_dir, 'pca_visualization.html'))
    print("PCA visualization saved as 'pca_visualization.html'")
    print("Open this file in your browser to view the interactive plot")

if __name__ == "__main__":
    print("Loading dataset...")
    train_dataset, test_dataset, weights = load_train_test_dataset(csv_path="/Users/laura/Documents/aura/tuh_ecg_features2.csv")
    
    # Reduce sample size for faster computation
    sampled_indices = random.sample(list(range(len(train_dataset))), 2000)
    print(f"Using {len(sampled_indices)} samples for visualization")

    # Create subset
    train_subset = Subset(train_dataset, sampled_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    print("Loading model...")
    model = DeepResidualEmbeddingModel(input_dim=14, embedding_dim=1024)
    model.load_state_dict(torch.load('checkpoints/model_9.pth'))
    model.eval()  # Set to evaluation mode

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    print("Generating embeddings...")
    encoded_data = []
    labels = []
    patient_ids = []
    with torch.no_grad():
        for features, label, patient_id in tqdm(train_dataloader, desc="Processing batches"):
            features, label, patient_id = features.to(device), label.to(device), patient_id.to(device)
            embeddings = model(features)
            encoded_data.extend(embeddings.cpu().numpy())
            labels.extend(label.cpu().numpy())
            patient_ids.extend(patient_id.cpu().numpy())
    
    # Convert lists to numpy arrays
    encoded_data = np.array(encoded_data)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    print(f"Generated embeddings shape: {encoded_data.shape}")

    print("Creating visualizations...")
    results_dir = f'results_{time.time()}'
    os.makedirs(results_dir, exist_ok=True)
    visualize_embeddings(encoded_data,  patient_ids, results_dir)
    visualize_umap(encoded_data, labels, patient_ids, results_dir)
    print("All visualizations completed!")