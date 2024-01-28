import torch
from torch import nn

from ..utils import distance


class TransE(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(TransE, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.output_dim = dim
        self.device = device
        self.entities = nn.Embedding(entity_size, dim)
        self.relations = nn.Embedding(relation_size, dim)
        nn.init.xavier_normal_(self.entities.weight)
        nn.init.xavier_normal_(self.relations.weight)
        self.criterion = nn.MarginRankingLoss(margin=1.0, reduction='none')


    def forward(self, positive_triples, negative_triples):
        positive_distances = self._kg_triplet_distance(positive_triples)
        negative_distances = self._kg_triplet_distance(negative_triples)
        loss = self._loss(positive_distances, negative_distances).mean()
        return loss


    def _kg_triplet_distance(self, triples):
        heads = triples[:, 0] # batch
        relations = triples[:, 1] # batch
        tails = triples[:, 2] # batch
        heads = self.entities(heads) # batch x dim
        relations = self.relations(relations) # batch x dim
        tails = self.entities(tails) # batch x dim
        return distance(torch.stack([heads, relations, tails], dim=1))


    def _loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)
