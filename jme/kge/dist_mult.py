import torch
from torch import nn


class DistMult(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(DistMult, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.device = device

        self.entities = nn.Embedding(entity_size, dim)
        self.relations = nn.Embedding(relation_size, dim)

        nn.init.xavier_normal_(self.entities.weight)
        nn.init.xavier_normal_(self.relations.weight)

        self.criterion = nn.MarginRankingLoss(margin=1.0)


    def forward(self, positive_triples, negative_triples):
        positive_scores = self._calc_score(positive_triples)
        negative_scores = self._calc_score(negative_triples)
        loss = self._loss(positive_scores, negative_scores)
        return loss


    def _calc_score(self, triples):
        heads = triples[:, 0] # batch
        relations = triples[:, 1] # batch
        tails = triples[:, 2] # batch

        h = self.entities(heads) # batch x dim
        r = self.relations(relations) # batch x dim
        t = self.entities(tails) # batch x dim

        score = (h * r) * t
        score = torch.sum(score, -1).flatten()
        return score


    def _loss(self, positive_scores, negative_scores):
        target = torch.ones(positive_scores.shape[0], device=self.device) * -1
        return self.criterion(positive_scores, negative_scores, target)
