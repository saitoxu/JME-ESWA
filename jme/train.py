import os
from time import time

import torch
from torch.utils.data import DataLoader

from .dataset import KGDataset, Phase, RecDataset, ValOrTestDataset
from .jme import JME
from .logger import getLogger
from .parser import parse_args
from .utils import EarlyStopping, evaluate, log_results, seed_everything


def train(train_rec_dataloader, train_kg_dataloader, model, optimizer, args, device, logger):
    rec_size = len(train_rec_dataloader.dataset)
    model.train()

    kg_size = len(train_kg_dataloader.dataset)
    size = min(rec_size, kg_size)
    for batch, (rec_data, kg_data) in enumerate(zip(train_rec_dataloader, train_kg_dataloader)):
        u, i, j, interactions = rec_data
        u, i, j, interactions = u.to(device), i.to(device), j.to(device), interactions.to(device)

        positive_triples, negative_triples = kg_data
        positive_triples, negative_triples = positive_triples.to(device), negative_triples.to(device)

        model.normalize()

        loss = model((u, i, j, interactions), (positive_triples, negative_triples))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(u)
            logger.debug(f"Loss: {loss:>7f} [{current:>8d}/{size:>8d}]")


def validate(dataloader, model, Ks, device, logger):
    mrr, hrs, ndcgs = evaluate(dataloader, model, Ks, device)
    log_results(mrr, hrs, ndcgs, logger.debug)
    return hrs[0]


if __name__ == '__main__':
    start = int(time())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    logger = getLogger(name=__name__, path=args.save_path)

    for key, value in vars(args).items():
        logger.debug(f'{key}: {value}')

    torch.backends.cudnn.benchmark = True

    data_path = f"dataset/{args.dataset}"
    behavior_data = eval(args.behavior_data)
    num_workers = 2 if os.cpu_count() > 1 else 0

    train_rec_data = RecDataset(data_path=data_path, behavior_data=behavior_data, neg_size=args.neg_size)
    train_rec_dataloader = DataLoader(train_rec_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    train_kg_data = KGDataset(data_path=data_path, neg_size=args.neg_size)
    train_kg_dataloader = DataLoader(train_kg_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)

    val_data = ValOrTestDataset(data_path, phase=Phase.VAL, train_data=behavior_data)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    user_entity_map = torch.tensor(train_kg_data.user_entity_map).to(device)
    item_entity_map = torch.tensor(train_kg_data.item_entity_map).to(device)

    model = JME(
        entity_size=train_kg_data.entity_size,
        relation_size=train_kg_data.relation_size,
        user_size=train_rec_data.user_size,
        item_size=train_rec_data.item_size,
        behavior_size=train_rec_data.behavior_size,
        dim=args.dim,
        user_entity_map=user_entity_map,
        item_entity_map=item_entity_map,
        use_boac=args.use_boac,
        use_bam=args.use_bam,
        use_mbl=args.use_mbl,
        use_epl=args.use_epl,
        user_masters=user_masters,
        job_masters=job_masters,
        user_consistencies=user_consistencies,
        use_csw=args.use_csw,
        consistency_weight=args.consistency_weight,
        kge=args.kge,
        device=device
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(args.patience)

    Ks = eval(args.Ks)
    epoch = args.epoch
    eval_interval = 5
    for t in range(args.epoch):
        logger.debug(f"Epoch {t+1}")
        logger.debug('-'*32)
        train(train_rec_dataloader, train_kg_dataloader, model, optimizer, args, device, logger)
        torch.save(model, args.save_path + 'latest.pth')
        if (t+1) % eval_interval == 0:
            hr = validate(val_dataloader, model, Ks, device, logger)
            # early stopping
            should_save, should_stop = early_stop(hr)
            if should_save:
                torch.save(model, args.save_path + 'best.pth')
            if should_stop:
                epoch = t + 1
                logger.debug('Early stopping.')
                break
    end = int(time())
    logger.debug('Done!')
