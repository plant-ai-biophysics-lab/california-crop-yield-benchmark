import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size

        device = torch.device(device)
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", 
                             (~torch.eye(batch_size * 2, 
                                         batch_size * 2, 
                                         dtype=bool)).float().to(device))
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss



# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.5):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, features, batch_size):
#         cos_sim = torch.matmul(features, features.T) / self.temperature
#         labels = torch.range(batch_size).repeat(2) # Assuming pairs are arranged consecutively
#         mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).T)
#         positive_samples = cos_sim.masked_select(mask).view(batch_size, -1)
#         negative_samples = cos_sim.masked_select(~mask).view(batch_size, -1)
#         losses = torch.cat([torch.logsumexp(negative_samples, dim=1), -torch.diag(positive_samples)], dim=0)
#         return losses.mean()
    

    