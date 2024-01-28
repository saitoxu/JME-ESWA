import random
from collections import defaultdict
from enum import Enum

import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, data_path: str, behavior_data: list, neg_size: int):
        self.user_size, self.item_size = calc_max_ids(data_path)
        self.behavior_data = [{} for _ in range(len(behavior_data))]
        self.behavior_size = len(behavior_data)
        self.neg_size = neg_size
        self.data = []
        for idx, file_name in enumerate(behavior_data):
            with open(f'{data_path}/{file_name}') as f:
                for line in f:
                    user_id, *item_ids = list(map(lambda x: int(x), line.split(' ')))
                    self.behavior_data[idx][user_id] = item_ids

        for user_id in range(self.user_size):
            observed = self._observed_items(user_id)
            for item_id in observed:
                interactions = []
                for b in self.behavior_data:
                    interaction = int(item_id in b.get(user_id, []))
                    interactions.append(interaction)
                self.data.append([user_id, item_id, torch.tensor(interactions)])
        self.data = self.data * self.neg_size

        self.all_item_ids = set([x for x in range(self.item_size)])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        u, i, interactions = self.data[idx]
        observed = self._observed_items(u)
        j = self._negative_sampling(observed)
        return u, i, j, interactions


    def _observed_items(self, user_id):
        observed = []
        for b in self.behavior_data:
            observed += b.get(user_id, [])
        return set(observed)


    def _negative_sampling(self, observed_item_ids):
        return random.choice(list(self.all_item_ids - observed_item_ids))


class KGDataset(Dataset):
    def __init__(self, data_path: str, neg_size: int):
        self.entity_size = 0
        self.relation_size = 0
        self.neg_size = neg_size
        self.data = []
        self.user_entity_map = []
        self.item_entity_map = []
        with open(data_path + '/kg.txt') as f:
            for line in f:
                h, r, t = list(map(lambda x: int(x), line.split(' ')))
                self.data.append([h, r, t])
                self.entity_size = max(self.entity_size, h + 1, t + 1)
                self.relation_size = max(self.relation_size, r + 1)
        with open(data_path + '/user_entity_map.txt') as f:
            for line in f:
                e, u = list(map(lambda x: int(x), line.split(' ')))
                self.user_entity_map.append([u, e])
        with open(data_path + '/item_entity_map.txt') as f:
            for line in f:
                e, i = list(map(lambda x: int(x), line.split(' ')))
                self.item_entity_map.append([i, e])
        self.data = self.data * self.neg_size


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        positive_triples = self.data[idx]
        h, r, t = positive_triples
        head_or_tail = random.randint(0, 1)
        neg_h = h if head_or_tail == 0 else random.randint(0, self.entity_size - 1)
        neg_t = t if head_or_tail == 1 else random.randint(0, self.entity_size - 1)
        negative_triples = [neg_h, r, neg_t]
        return torch.tensor(positive_triples), torch.tensor(negative_triples)


class Phase(str, Enum):
    VAL = 'val'
    TEST = 'test'


class ValOrTestDataset(Dataset):
    def __init__(self, data_path: str, phase: Phase, train_data: list):
        self.phase = phase
        self.user_size, self.item_size = calc_max_ids(data_path)
        self.users = defaultdict(set)
        self.data = []
        for name in train_data:
            with open(f'{data_path}/{name}') as f:
                for line in f:
                    user_id, *item_ids = list(map(lambda x: int(x), line.split(' ')))
                    self.users[user_id] |= set(item_ids)
        with open(data_path + '/val.txt') as f:
            for line in f:
                user_id, item_id = list(map(lambda x: int(x), line.split(' ')))
                self.user_size = max(self.user_size, user_id + 1)
                self.item_size = max(self.item_size, item_id + 1)
                if self.phase == 'val':
                    self.data.append([user_id, item_id])
        with open(data_path + '/test.txt') as f:
            for line in f:
                user_id, item_id = list(map(lambda x: int(x), line.split(' ')))
                self.user_size = max(self.user_size, user_id + 1)
                self.item_size = max(self.item_size, item_id + 1)
                if self.phase == 'test':
                    self.data.append([user_id, item_id])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        user_id, item_id = self.data[idx]
        observed_item_ids = set(self.users[user_id])
        all_item_ids = set([x for x in range(self.item_size)])
        candidate_neg_items = all_item_ids - observed_item_ids - set([item_id])
        negative_item_ids = random.sample(list(candidate_neg_items), min(len(candidate_neg_items), 99))
        dummy_ids = [0] * (self.item_size - len(negative_item_ids) - 1)
        negative_item_ids = dummy_ids + negative_item_ids
        return user_id, item_id, torch.tensor(negative_item_ids), len(dummy_ids)


def calc_max_ids(data_path: str):
    user_size = 0
    item_size = 0
    all_behavior_data = ['train_view.txt', 'train_fav.txt', 'train.txt']
    for file_name in all_behavior_data:
        with open(f'{data_path}/{file_name}') as f:
            for line in f:
                user_id, *item_ids = list(map(lambda x: int(x), line.split(' ')))
                user_size = max(user_size, user_id + 1)
                item_size = max(item_size, max(item_ids) + 1)
    return user_size, item_size


def load_masters(data_path: str, device: str):
    results = []
    for fname in ["user_masters.txt", "item_masters.txt"]:
        result = []
        with open(f"{data_path}/{fname}") as f:
            for line in f:
                _, *masters = list(map(lambda x: int(x), line.split(" ")))
                result.append(masters)
        results.append(torch.tensor(result, dtype=torch.int, device=device))
    return results


def load_user_consistencies(data_path: str, device: str):
    behavior_data = ['train_view.txt', 'train_fav.txt']

    user_masters, job_masters = _load_masters(data_path)
    user_interactions = [set() for _ in range(len(user_masters))]
    user_similarities = [[] for _ in range(len(user_masters))]
    tfidf_matrix = []
    for file_name in behavior_data:
        with open(f'{data_path}/{file_name}') as f:
            for line in f:
                user_id, *item_ids = list(map(lambda x: int(x), line.split(' ')))
                user_master = user_masters[user_id]
                for item_id in item_ids:
                    if item_id in user_interactions[user_id]:
                        continue
                    user_interactions[user_id].add(item_id)
                    job_master = job_masters[item_id]
                    similarity = len(user_master & job_master) / len(user_master | job_master)
                    user_similarities[user_id].append(similarity)
    consistencies = []
    for user_id in range(len(user_masters)):
        avg_similarity = sum(user_similarities[user_id]) / len(user_similarities[user_id])
        consistencies.append(avg_similarity)
    return torch.tensor(consistencies, dtype=torch.float, device=device), torch.tensor(tfidf_matrix, dtype=torch.float, device=device)


def _load_masters(data_path: str):
    results = []
    for fname in ["user_masters.txt", "item_masters.txt"]:
        result = defaultdict(set)
        with open(f"{data_path}/{fname}") as f:
            for line in f:
                user_id, *masters = list(map(lambda x: int(x), line.split(" ")))
                for i, master in enumerate(masters):
                    if master == 1:
                        result[user_id].add(i)
        results.append(result)
    return results
