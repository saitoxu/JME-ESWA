import torch
from torch import nn


class ComplEx(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(ComplEx, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.output_dim = dim * 2
        self.device = device

        self.entities_re = nn.Embedding(entity_size, dim)
        self.entities_im = nn.Embedding(entity_size, dim)
        self.relations_re = nn.Embedding(relation_size, dim)
        self.relations_im = nn.Embedding(relation_size, dim)

        nn.init.xavier_normal_(self.entities_re.weight)
        nn.init.xavier_normal_(self.entities_im.weight)
        nn.init.xavier_normal_(self.relations_re.weight)
        nn.init.xavier_normal_(self.relations_im.weight)

        self.criterion = nn.MarginRankingLoss(margin=1.0)


    def forward(self, positive_triples, negative_triples):
        positive_scores = self._calc_score(positive_triples)
        negative_scores = self._calc_score(negative_triples)
        loss = self._loss(positive_scores, negative_scores)
        _lambda = 0.0001
        return loss + _lambda * self._regularization()


    def entities(self, e_batch):
        e_re = self.entities_re(e_batch)
        e_im = self.entities_im(e_batch)
        return torch.cat([e_re, e_im], dim=1)


    def relations(self, r_batch):
        r_re = self.relations_re(r_batch)
        r_im = self.relations_im(r_batch)
        return torch.cat([r_re, r_im], dim=1)


    def _calc_score(self, triples):
        heads = triples[:, 0] # batch
        relations = triples[:, 1] # batch
        tails = triples[:, 2] # batch

        h_re = self.entities_re(heads) # batch x dim
        h_im = self.entities_im(heads) # batch x dim
        r_re = self.relations_re(relations) # batch x dim
        r_im = self.relations_im(relations) # batch x dim
        t_re = self.entities_re(tails) # batch x dim
        t_im = self.entities_im(tails) # batch x dim

        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )


    def _loss(self, positive_scores, negative_scores):
        scores = torch.cat([positive_scores, -negative_scores])
        return torch.log(torch.exp(-scores) + 1).mean()


    def _regularization(self):
        e_re = torch.norm(self.entities_re.weight, p=2)
        e_im = torch.norm(self.entities_im.weight, p=2)
        r_re = torch.norm(self.relations_re.weight, p=2)
        r_im = torch.norm(self.relations_im.weight, p=2)
        return e_re + e_im + r_re + r_im
