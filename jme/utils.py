import os
import random

import numpy as np
import torch

from .metrics import calc_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.best_value = .0
        self.count = 0
        self.patience = patience


    def __call__(self, value: float):
        should_save, should_stop = False, False
        if value > self.best_value:
            self.best_value = value
            self.count = 0
            should_save = True
        else:
            self.count += 1
        if self.count >= self.patience:
            should_stop = True
        return should_save, should_stop


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(dataloader, model, Ks, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for u, i, js, dummy_counts in dataloader:
            for idx in range(len(u)):
                dummy_cnt = dummy_counts[idx]
                user = u[idx].to(device)
                items = torch.cat((i.reshape(-1, 1), js), 1).to(device)
                pred = model.predict(user, items[idx])
                pred = torch.cat((pred[0].reshape(-1), pred[dummy_cnt+1:])).cpu()
                preds.append(pred)
    return calc_metrics(preds, Ks)


def log_results(mrr, hrs, ndcgs, log=print):
    rounded_hrs = list(map(lambda x: float(f'{x:>7f}'), hrs))
    rounded_ndcgs = list(map(lambda x: float(f'{x:>7f}'), ndcgs))
    log(f'MRR:\t{mrr:>7f}')
    log(f'HRs:\t{rounded_hrs}')
    log(f'NDCGs:\t{rounded_ndcgs}')


def distance(triplets):
    assert triplets.size()[1] == 3
    heads = triplets[:, 0, :]
    relations = triplets[:, 1, :]
    tails = triplets[:, 2, :]
    return (heads + relations - tails).norm(dim=1)
