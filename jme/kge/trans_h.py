import torch
import torch.nn.functional as F
from torch import nn


class TransH(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(TransH, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.output_dim = dim
        self.device = device
        self.entities = nn.Embedding(entity_size, dim)
        self.relations = nn.Embedding(relation_size, dim)
        self.norm_vectors = nn.Embedding(relation_size, dim)
        nn.init.xavier_normal_(self.entities.weight)
        nn.init.xavier_normal_(self.relations.weight)
        nn.init.xavier_normal_(self.norm_vectors.weight)
        self.pdist = nn.PairwiseDistance()


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
        w_r = self.norm_vectors(relations) # batch x dim

        h = self._transfer(h, w_r)
        t = self._transfer(t, w_r)

        return self.pdist(h + r, t) ** 2


    def _transfer(self, e, norm):
        norm = F.normalize(norm, p = 2, dim = -1)
        return e - torch.sum(e * norm, -1, True) * norm


    def _loss(self, positive_scores, negative_scores):
        C = 1.0
        margin = 1.0
        eps = 0.001
        margin_loss = F.relu(positive_scores + margin - negative_scores).mean()
        scale_loss = F.relu(torch.norm(self.entities.weight, p=2, dim=1, keepdim=False) - 1.0).mean()
        w_r = self.norm_vectors.weight
        d_r = self.relations.weight
        wr_dr = torch.sum((w_r * d_r), dim=1, keepdim=False)
        dr2 = torch.norm(d_r, p=2, dim=1, keepdim=False)
        orth_loss = F.relu(torch.sum((wr_dr / dr2)**2 - eps**2)).mean()

        return margin_loss + C * (scale_loss + orth_loss)
