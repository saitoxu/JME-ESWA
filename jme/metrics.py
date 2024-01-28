import torch
import numpy as np


def calc_metrics(preds, Ks):
    ranks = []
    for pred in preds:
        _, indices = torch.sort(pred, descending=True)
        rank = indices.argmin() + 1
        ranks.append(rank.item())
    return mrr(ranks), hit_ratios(ranks, Ks), ndcgs(ranks, Ks)


def mrr(ranks):
    return np.array(list(map(lambda x: 1 / x, ranks))).sum() / len(ranks)


def hit_ratios(ranks, Ks):
    results = []
    for k in Ks:
        hr = len(list(filter(lambda x: x <= k, ranks))) / len(ranks)
        results.append(hr)
    return results


def ndcgs(ranks, Ks):
    results = []
    for k in Ks:
        ndcg = np.array(list(map(lambda x: 1 / np.log2(x + 1), list(filter(lambda x: x <= k, ranks))))).sum() / len(ranks)
        results.append(ndcg)
    return results
