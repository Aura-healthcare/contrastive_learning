import torch.nn as nn
import torch.nn.functional as F
import torch

# The ideal distance metric for a positive sample is set to 1, for a negative sample it is set to 0      
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, contrastive, label):
        # label: 1 for positive pair, 0 for negative pair
        # map to {-1, +1} to match cosine similarity range
        target = 2 * label - 1
        score = self.similarity(anchor, contrastive)
        return ((score - target) ** 2).mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).

    For each anchor, ALL samples sharing the same label are treated as positives
    (across any patient). All samples with a different label are negatives.

    Loss per anchor i:
        L_i = -1/|P(i)| * sum_{p in P(i)} log(
                exp(sim(i,p) / τ) / sum_{a ≠ i} exp(sim(i,a) / τ)
              )

    Embeddings are L2-normalised internally so only their direction matters.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings : (B, D)
        labels     : (B,)  — seizure/non-seizure class labels
        """
        device = embeddings.device
        B = embeddings.size(0)

        emb = F.normalize(embeddings, p=2, dim=1)          # (B, D)
        sim = torch.matmul(emb, emb.T) / self.temperature  # (B, B)

        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask

        # Numerical stability: subtract row max before exp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim  = torch.exp(sim) * ~self_mask              # exclude self from denominator
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.sum(dim=1).float()
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)
        return loss[valid].mean()


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