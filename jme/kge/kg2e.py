import torch
import torch.nn as nn
import torch.nn.functional as F


class KG2E(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str, margin=1.0):
        super(KG2E, self).__init__()
        self.device = device
        self.margin = margin
        self.ke = dim

        self.entity_emb = nn.Embedding(num_embeddings=entity_size, embedding_dim=dim)
        self.entity_covar = nn.Embedding(num_embeddings=entity_size, embedding_dim=dim)
        self.relation_emb = nn.Embedding(num_embeddings=relation_size, embedding_dim=dim)
        self.relation_covar = nn.Embedding(num_embeddings=relation_size, embedding_dim=dim)
        nn.init.xavier_normal_(self.entity_emb.weight)
        nn.init.xavier_normal_(self.entity_covar.weight)
        nn.init.xavier_normal_(self.relation_emb.weight)
        nn.init.xavier_normal_(self.relation_covar.weight)


    def kl_score(self, relation_m, relation_v, error_m, error_v):
        eps = 1e-9
        ep1_er = torch.sum(error_v / (relation_v + eps), dim=1)
        ep2_er = torch.sum((relation_m-error_m)**2 / (relation_v + eps), dim=1)
        kl_er = (ep1_er + ep2_er - self.ke) / 2

        ep1_re = torch.sum(relation_v / (error_v + eps), dim=1)
        ep2_re = torch.sum((error_m - relation_m) ** 2 / (error_v + eps), dim=1)
        kl_re = (ep1_re + ep2_re - self.ke) / 2
        return (kl_er + kl_re) / 2


    def score(self, triples):
        head, relation, tail = torch.chunk(input=triples, chunks=3, dim=1)
        head_m = torch.squeeze(self.entity_emb(head), dim=1)
        head_v = torch.squeeze(self.entity_covar(head), dim=1)
        tail_m = torch.squeeze(self.entity_emb(tail), dim=1)
        tail_v = torch.squeeze(self.entity_covar(tail), dim=1)
        relation_m = torch.squeeze(self.relation_emb(relation), dim=1)
        relation_v = torch.squeeze(self.relation_covar(relation), dim=1)
        error_m = tail_m - head_m
        error_v = tail_v + head_v
        return self.kl_score(relation_m, relation_v, error_m, error_v)


    def normalize(self):
        ee = self.entity_emb
        re = self.relation_emb
        ec = self.entity_covar
        rc = self.relation_covar
        ee.weight.data.copy_(torch.renorm(input=ee.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        re.weight.data.copy_(torch.renorm(input=re.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        ec.weight.data.copy_(torch.renorm(input=ec.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))
        rc.weight.data.copy_(torch.renorm(input=rc.weight.detach().cpu(), p=2, dim=0, maxnorm=1.0))


    def forward(self, pos, neg):
        size = pos.size()[0]
        pos_score = self.score(pos)
        neg_score = self.score(neg)
        return torch.sum(F.relu(input=pos_score-neg_score+self.margin)) / size


    def entities(self, e_batch):
        ee = self.entity_emb(e_batch)
        ec = self.entity_covar(e_batch)
        return torch.cat([ee, ec], dim=1)


    def relations(self, r_batch):
        re = self.relation_emb(r_batch)
        rc = self.relation_covar(r_batch)
        return torch.cat([re, rc], dim=1)
