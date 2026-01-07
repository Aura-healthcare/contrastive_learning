import torch.nn as nn
import torch.nn.functional as F
import torch

# The ideal distance metric for a positive sample is set to 1, for a negative sample it is set to 0      
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
        self.temperature = temperature

    def forward(self, anchor, contrastive, distance):
        # use cosine similarity from torch to get score
        score = self.similarity(anchor, contrastive)
        # Apply temperature scaling to make the loss more sensitive
        score = score / self.temperature
        # after cosine apply MSE between distance and score
        return nn.MSELoss()(score, distance) #Ensures that the calculated score is close to the ideal distance (1 or 0)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: Tensor of shape (B, D)
        labels: Tensor of shape (B,) with int labels
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        loss = []
        for i in range(batch_size):
            anchor_label = labels[i]

            # Mask for positives and negatives
            is_pos = labels == anchor_label
            is_neg = labels != anchor_label

            # Remove self-comparison
            is_pos[i] = False

            # Get hardest positive (max dist)
            if torch.any(is_pos):
                hardest_pos = dist_matrix[i][is_pos].max()
            else:
                continue  # skip if no positive

            # Get hardest negative (min dist)
            if torch.any(is_neg):
                hardest_neg = dist_matrix[i][is_neg].min()
            else:
                continue  # skip if no negative

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss.append(triplet_loss)

        if len(loss) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return torch.stack(loss).mean()