import torch
import torch.nn.functional as F
from torch import nn


class ConvE(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, dim: int, device: str):
        super(ConvE, self).__init__()
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.dim = dim
        self.output_dim = dim
        self.device = device
        self.entities = nn.Embedding(entity_size, dim)
        self.relations = nn.Embedding(relation_size, dim)
        self.inp_drop = nn.Dropout(0.2)
        self.hidden_drop = nn.Dropout(0.3)
        self.feature_map_drop = nn.Dropout2d(0.2)
        self.loss = nn.BCELoss()
        self.emb_dim1 = dim // 8
        self.emb_dim2 = dim // self.emb_dim1
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(dim)
         # working only under dim = 64
        self.fc = nn.Linear(2688, dim)
        nn.init.xavier_normal_(self.entities.weight.data)
        nn.init.xavier_normal_(self.relations.weight.data)


    def forward(self, positive_triples, negative_triples):
        p_scores = self._calc_score(positive_triples)
        n_scores = self._calc_score(negative_triples)

        p_t = torch.ones(p_scores.shape, dtype=torch.float, device=self.device)
        n_t = torch.zeros(p_scores.shape, dtype=torch.float, device=self.device)

        scores = torch.cat([p_scores, n_scores])
        t = torch.cat([p_t, n_t])

        return self.loss(scores, t)


    def _calc_score(self, triples):
        e_s = triples[:, 0] # batch
        rel = triples[:, 1] # batch
        e_o = triples[:, 2] # batch
        e1_embedded= self.entities(e_s).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.relations(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = (x * self.entities(e_o)).sum(1, keepdim=True).view(-1)
        scores = torch.sigmoid(x)
        return scores
