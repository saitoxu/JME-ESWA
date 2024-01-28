import torch
from torch import nn

from .kge.trans_e import TransE
from .kge.trans_h import TransH
from .kge.trans_r import TransR
from .kge.dist_mult import DistMult
from .kge.compl_ex import ComplEx
from .kge.kg2e import KG2E
from .kge.conv_e import ConvE
from .utils import distance, MyTripletMarginWithDistanceLoss


class JME(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, user_size: int, item_size: int, \
            behavior_size: int, dim: int, user_entity_map: torch.tensor, item_entity_map: torch.tensor, \
            use_boac: int, use_bam: int, use_epl: int, kge: str, device: str):
        super(JME, self).__init__()
        self.use_epl = use_epl == 1
        self.use_bam = use_bam == 1
        self.use_boac = use_boac == 1
        self.user_size = user_size
        self.item_size = item_size
        self.behavior_size = behavior_size
        self.device = device
        self.user_entity_map = user_entity_map
        self.item_entity_map = item_entity_map

        mbl_relation_size = behavior_size
        if self.use_boac:
            mbl_relation_size = 2**behavior_size - 1

        KGE = kge_class(kge)
        if self.use_epl:
            self.epl_module = KGE(
                entity_size=entity_size,
                relation_size=relation_size,
                dim=dim,
                device=device
            )
        self.mbl_module = KGE(
            entity_size=user_size + item_size,
            relation_size=mbl_relation_size,
            dim=dim,
            device=device
        )

        self.triplet_loss = MyTripletMarginWithDistanceLoss(distance_function=self._distance_function)
        self.sigmoid = nn.Sigmoid()


    def _distance_function(self, x, y):
        r = torch.zeros(x.shape, dtype=torch.long, device=self.device)
        triplets = torch.stack([x, r, y], dim=1)
        return distance(triplets)


    def forward(self, behavior_data, kg_data):
        u_batch, i_batch, j_batch, interactions = behavior_data
        mb_u_batch = self._resolve_user_indexes_mbl(u_batch)
        mb_i_batch = self._resolve_item_indexes_mbl(i_batch)
        mb_j_batch = self._resolve_item_indexes_mbl(j_batch)
        mb_b_batch = self._interaction_indexes(interactions)
        mb_positive_triples = torch.stack([mb_u_batch, mb_b_batch, mb_i_batch], dim=1)
        mb_negative_triples = torch.stack([mb_u_batch, mb_b_batch, mb_j_batch], dim=1)
        mb_loss = self.mbl_module(mb_positive_triples, mb_negative_triples)

        u = self.mbl_module.entities(mb_u_batch)
        i = self.mbl_module.entities(mb_i_batch)
        j = self.mbl_module.entities(mb_j_batch)

        loss = mb_loss

        if self.use_epl:
            positive_triples, negative_triples = kg_data
            ep_loss = self.epl_module(positive_triples, negative_triples)

            ep_u_batch = self._resolve_user_indexes_epl(u_batch)
            ep_i_batch = self._resolve_item_indexes_epl(i_batch)
            ep_j_batch = self._resolve_item_indexes_epl(j_batch)
            u_entities = self.epl_module.entities(ep_u_batch)
            i_entities = self.epl_module.entities(ep_i_batch)
            j_entities = self.epl_module.entities(ep_j_batch)

            u += u_entities
            i += i_entities
            j += j_entities

            loss += ep_loss

        m = self._rec_margin(mb_b_batch)

        rec_loss = self.triplet_loss(u, i, j, m)
        loss += rec_loss
        return loss


    def predict(self, u_idx, items):
        mb_u_batch = self._resolve_user_indexes_mbl(u_idx.reshape(1))
        mb_i_batch = self._resolve_item_indexes_mbl(items)
        u = self.mbl_module.entities(mb_u_batch).reshape(-1)
        i = self.mbl_module.entities(mb_i_batch)

        if self.use_epl:
            ep_u_batch = self._resolve_user_indexes_epl(u_idx.reshape(1))
            ep_i_batch = self._resolve_item_indexes_epl(items)
            u_entities = self.epl_module.entities(ep_u_batch).reshape(-1)
            i_entities = self.epl_module.entities(ep_i_batch)

            u += u_entities
            i += i_entities

        item_size = i.size()[0]
        u = u.repeat(item_size).reshape(item_size, -1)
        r = torch.zeros(u.shape, dtype=torch.long, device=self.device)
        triplets = torch.stack([u, r, i], dim=1)
        scores = -distance(triplets)
        return scores


    def normalize(self):
        method = 'normalize'
        modules = [self.mbl_module]
        if self.use_epl:
            modules.append(self.epl_module)
        for module in modules:
            if method in dir(module) and callable(getattr(module, method)):
                module.normalize()


    def _interaction_indexes(self, interactions):
        """
        args:
            interactions: batch x behavior_size (e.g. view, fav, apply)
        return:
            interaction indexes: batch
        """
        batch = len(interactions)
        bcs = torch.zeros(batch, dtype=torch.int, device=self.device)
        if self.use_boac:
            for i in range(self.behavior_size):
                bcs += interactions[:, i] * 2**i
            bcs -= 1
        else:
            for i in range(self.behavior_size):
                bcs, _ = torch.max(torch.stack([bcs, interactions[:, i] * i]), dim=0)
        return bcs


    def _resolve_user_indexes_mbl(self, u_batch):
        return u_batch


    def _resolve_item_indexes_mbl(self, i_batch):
        return i_batch + self.user_size


    def _resolve_user_indexes_epl(self, u_batch):
        return self.user_entity_map[u_batch, :][:, 1]


    def _resolve_item_indexes_epl(self, i_batch):
        return self.item_entity_map[i_batch, :][:, 1]


    def _rec_margin(self, mb_b_batch):
        if self.use_bam:
            norms = self.mbl_module.relations(mb_b_batch).norm(dim=1)
            alpha = 1.5
            return alpha - self.sigmoid(norms)
        return 1.0


def kge_class(method_name):
    assert method_name in ['trans_e', 'trans_h', 'trans_r', 'dist_mult', 'compl_ex', 'kg2e', 'conv_e']
    if method_name == 'trans_e':
        return TransE
    elif method_name == 'trans_h':
        return TransH
    elif method_name == 'trans_r':
        return TransR
    elif method_name == 'dist_mult':
        return DistMult
    elif method_name == 'compl_ex':
        return ComplEx
    elif method_name == 'kg2e':
        return KG2E
    else:
        return ConvE
