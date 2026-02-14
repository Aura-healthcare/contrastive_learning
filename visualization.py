import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from tqdm import tqdm
from model import EmbeddingModel, DeepResidualEmbeddingModel
import plotly.graph_objects as go
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
import random
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


def plot_evaluation_results(metrics, predictions_probs, true_labels, save_path='./evaluation_results.png'):
    """
    Create a combined visualization showing ROC curve and confusion matrix.

    Args:
        metrics: Dictionary containing evaluation metrics including 'roc_auc' and 'confusion_matrix'
        predictions_probs: Array of predicted probabilities for the positive class
        true_labels: Array of true labels
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ROC Curve
    if metrics['roc_auc'] is not None:
        fpr, tpr, thresholds = roc_curve(true_labels, predictions_probs)

        axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'ROC curve not available\n(single class present)',
                     ha='center', va='center', fontsize=12)
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')

    # Plot 2: Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, ax=axes[1],
                annot_kws={'fontsize': 14})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['No Seizure', 'Seizure'])
    axes[1].set_yticklabels(['No Seizure', 'Seizure'], rotation=0)

    # Add metrics text
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall: {metrics['recall']:.3f}\n"
        f"F1 Score: {metrics['f1']:.3f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Evaluation results saved to {save_path}")
    return save_path


def plot_training_history(train_history, save_path='./training_history.png'):
    """
    Plot training loss and accuracy over epochs.

    Args:
        train_history: Dictionary with 'loss' and 'accuracy' lists
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_history['loss']) + 1)

    # Plot loss
    axes[0].plot(epochs, train_history['loss'], 'b-', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(epochs, train_history['accuracy'], 'g-', linewidth=2, marker='o')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training history saved to {save_path}")
    return save_path


# if __name__ == "__main__":
#     from app import load_train_test_dataset

#     print("Loading dataset...")
#     train_dataset, test_dataset, weights = load_train_test_dataset(csv_path="/Users/laura/Documents/aura/tuh_ecg_features2.csv")
    
#     # Reduce sample size for faster computation
#     sampled_indices = random.sample(list(range(len(train_dataset))), 2000)
#     print(f"Using {len(sampled_indices)} samples for visualization")

#     # Create subset
#     train_subset = Subset(train_dataset, sampled_indices)

#     train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#     print("Loading model...")
#     model = DeepResidualEmbeddingModel(input_dim=14, embedding_dim=1024)
#     model.load_state_dict(torch.load('checkpoints/model_9.pth'))
#     model.eval()  # Set to evaluation mode

#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')

#     print("Generating embeddings...")
#     encoded_data = []
#     labels = []
#     patient_ids = []
#     with torch.no_grad():
#         for features, label, patient_id in tqdm(train_dataloader, desc="Processing batches"):
#             features, label, patient_id = features.to(device), label.to(device), patient_id.to(device)
#             embeddings = model(features)
#             encoded_data.extend(embeddings.cpu().numpy())
#             labels.extend(label.cpu().numpy())
#             patient_ids.extend(patient_id.cpu().numpy())
    
#     # Convert lists to numpy arrays
#     encoded_data = np.array(encoded_data)
#     labels = np.array(labels)
#     patient_ids = np.array(patient_ids)
    
#     print(f"Generated embeddings shape: {encoded_data.shape}")

#     print("Creating visualizations...")
#     results_dir = f'results_{time.time()}'
#     os.makedirs(results_dir, exist_ok=True)
#     visualize_embeddings(encoded_data,  patient_ids, results_dir)
#     visualize_umap(encoded_data, labels, patient_ids, results_dir)
#     print("All visualizations completed!")