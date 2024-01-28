import torch
from torch import nn
import torch.nn.functional as F


class TransR(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(TransR, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.device = device

        self.entities = nn.Embedding(entity_size, dim)
        self.relations = nn.Embedding(relation_size, dim)
        self.transfer_matrix = nn.Embedding(self.relation_size, self.dim * self.dim)

        nn.init.xavier_normal_(self.entities.weight)
        nn.init.xavier_normal_(self.relations.weight)
        nn.init.xavier_normal_(self.transfer_matrix.weight)

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
        r_transfer = self.transfer_matrix(relations) # batch x (dim x dim)

        h = self._transfer(h, r_transfer)
        t = self._transfer(t, r_transfer)

        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)

        score = (h + r) - t
        score = torch.norm(score, 1, -1).flatten()
        return score


    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim, self.dim)
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], self.dim).permute(1, 0, 2)
            e = torch.matmul(e, r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.dim)
            e = torch.matmul(e, r_transfer)
        return e.view(-1, self.dim)


    def _loss(self, positive_scores, negative_scores):
        target = torch.ones(positive_scores.shape[0], device=self.device) * -1
        return self.criterion(positive_scores, negative_scores, target)
