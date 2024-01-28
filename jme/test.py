import torch
from torch.utils.data import DataLoader

from .parser import parse_args
from .dataset import ValOrTestDataset, Phase
from .utils import seed_everything, evaluate, log_results


def test(dataloader, model, Ks, device):
    mrr, hrs, ndcgs = evaluate(dataloader, model, Ks, device)
    log_results(mrr, hrs, ndcgs)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    data_path = f'dataset/{args.dataset}'
    all_behavior_data = ['train_view.txt', 'train_fav.txt', 'train.txt']

    test_data = ValOrTestDataset(data_path, phase=Phase.TEST, train_data=all_behavior_data)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    model = torch.load(args.model_path)

    Ks = eval(args.Ks)
    test(test_dataloader, model, Ks, device)
