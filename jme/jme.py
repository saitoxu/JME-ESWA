import torch
from torch import nn

from .kge.compl_ex import ComplEx
from .kge.conv_e import ConvE
from .kge.dist_mult import DistMult
from .kge.kg2e import KG2E
from .kge.trans_e import TransE
from .kge.trans_h import TransH
from .kge.trans_r import TransR
from .utils import distance


class JME(nn.Module):
    def __init__(self, entity_size: int, relation_size: int, user_size: int, item_size: int, \
            behavior_size: int, dim: int, user_entity_map: torch.tensor, item_entity_map: torch.tensor, \
            use_boac: int, use_bam: int, use_mbl: int, use_epl: int, use_csw: int, kge: str, device: str, \
            user_masters, item_masters, user_consistencies, consistency_weight: float):
        super(JME, self).__init__()
        self.use_mbl = use_mbl == 1
        self.use_epl = use_epl == 1
        self.use_bam = use_bam == 1
        self.use_boac = use_boac == 1
        self.use_csw = use_csw == 1
        self.user_size = user_size
        self.item_size = item_size
        self.behavior_size = behavior_size
        self.device = device
        self.user_entity_map = user_entity_map
        self.item_entity_map = item_entity_map
        self.user_masters = user_masters
        self.item_masters = item_masters
        self.user_consistencies = user_consistencies
        self.consistency_weight = consistency_weight

        mbl_relation_size = behavior_size
        if self.use_boac:
            mbl_relation_size = 2**behavior_size - 1

        KGE = kge_class(kge)
        self.output_dim = dim

        if self.use_epl:
            self.epl_module = KGE(
                entity_size=entity_size,
                relation_size=relation_size,
                dim=dim,
                device=device
            )
            self.output_dim = self.epl_module.output_dim

        if self.use_mbl:
            self.mbl_module = KGE(
                entity_size=user_size + item_size,
                relation_size=mbl_relation_size,
                dim=dim,
                device=device
            )
            self.output_dim = self.mbl_module.output_dim

        self.bam_linear = nn.Linear(self.output_dim, 1)

    def forward(self, behavior_data, kg_data):
        u_batch, i_batch, j_batch, interactions = behavior_data
        loss = 0.0
        if self.use_mbl:
            mb_u_batch = self._resolve_user_indexes_mbl(u_batch)
            mb_i_batch = self._resolve_item_indexes_mbl(i_batch)
            mb_j_batch = self._resolve_item_indexes_mbl(j_batch)
            mb_b_batch = self._interaction_indexes(interactions)
            mb_positive_triples = torch.stack([mb_u_batch, mb_b_batch, mb_i_batch], dim=1)
            mb_negative_triples = torch.stack([mb_u_batch, mb_b_batch, mb_j_batch], dim=1)
            mb_loss = self.mbl_module(mb_positive_triples, mb_negative_triples)
            mb_u = self.mbl_module.entities(mb_u_batch)
            mb_i = self.mbl_module.entities(mb_i_batch)
            mb_j = self.mbl_module.entities(mb_j_batch)
            loss += mb_loss

        if self.use_epl:
            positive_triples, negative_triples = kg_data
            ep_loss = self.epl_module(positive_triples, negative_triples)
            ep_u_batch = self._resolve_user_indexes_epl(u_batch)
            ep_i_batch = self._resolve_item_indexes_epl(i_batch)
            ep_j_batch = self._resolve_item_indexes_epl(j_batch)
            ep_u = self.epl_module.entities(ep_u_batch)
            ep_i = self.epl_module.entities(ep_i_batch)
            ep_j = self.epl_module.entities(ep_j_batch)
            loss += ep_loss

        if self.use_mbl and self.use_epl:
            ui_mbl_weights, ui_epl_weights = self._calc_weights(u_batch, i_batch)
            uj_mbl_weights, uj_epl_weights = self._calc_weights(u_batch, j_batch)
            ui = ui_mbl_weights * mb_u + ui_epl_weights * ep_u
            uj = uj_mbl_weights * mb_u + uj_epl_weights * ep_u
            i = ui_mbl_weights * mb_i + ui_epl_weights * ep_i
            j = uj_mbl_weights * mb_j + uj_epl_weights * ep_j
        elif self.use_mbl:
            ui = mb_u
            uj = mb_u
            i = mb_i
            j = mb_j
        elif self.use_epl:
            ui = ep_u
            uj = ep_u
            i = ep_i
            j = ep_j

        m = self._rec_margin(interactions)
        pos_dist = (ui - i).norm(dim=1)
        neg_dist = (uj - j).norm(dim=1)
        rec_loss = torch.clamp(pos_dist - neg_dist + m, min=0.0).mean()
        loss += rec_loss
        if self.use_bam:
            loss += torch.norm(self.bam_linear.weight, p=2)
        return loss

    def predict(self, u_idx, items):
        item_size = len(items)
        u_batch = u_idx.reshape(1).repeat(item_size)
        if self.use_mbl:
            mb_u_batch = self._resolve_user_indexes_mbl(u_batch)
            mb_i_batch = self._resolve_item_indexes_mbl(items)
            mb_u = self.mbl_module.entities(mb_u_batch)
            mb_i = self.mbl_module.entities(mb_i_batch)

        if self.use_epl:
            ep_u_batch = self._resolve_user_indexes_epl(u_batch)
            ep_i_batch = self._resolve_item_indexes_epl(items)
            ep_u = self.epl_module.entities(ep_u_batch)
            ep_i = self.epl_module.entities(ep_i_batch)

        if self.use_mbl and self.use_epl:
            ui_mbl_weights, ui_epl_weights = self._calc_weights(u_batch, items)
            ui = ui_mbl_weights * mb_u + ui_epl_weights * ep_u
            i = ui_mbl_weights * mb_i + ui_epl_weights * ep_i
        elif self.use_mbl:
            ui = mb_u
            i = mb_i
        elif self.use_epl:
            ui = ep_u
            i = ep_i

        r = torch.zeros(ui.shape, dtype=torch.long, device=self.device)
        triplets = torch.stack([ui, r, i], dim=1)
        scores = -distance(triplets)
        return scores

    def _calc_weights(self, users, items):
        if not self.use_csw:
            return 0.5, 0.5
        consistencies = self.user_consistencies[users]
        similarities = self._calc_similarities(users, items)
        epl_weights = self.consistency_weight * consistencies + (1 - self.consistency_weight) * similarities
        epl_weights = epl_weights.unsqueeze(dim=-1).repeat(1, self.output_dim)
        mbl_weights = 1 - epl_weights
        return mbl_weights, epl_weights

    def _calc_similarities(self, users, items):
        u_masters = self.user_masters[users, :]
        i_masters = self.item_masters[items, :]

        and_ui = (u_masters * i_masters)
        or_ui = (u_masters + i_masters) - and_ui
        jaccard_ui = and_ui.sum(dim=1) / or_ui.sum(dim=1)
        return jaccard_ui

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

    def _rec_margin(self, interactions):
        if self.use_mbl and self.use_bam:
            mb_b_batch = self._interaction_indexes(interactions)
            x = self.bam_linear(self.mbl_module.relations(mb_b_batch)).squeeze()
            return torch.sigmoid(x)
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
