#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np


def _get_targets_numpy(dataset):
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        return np.asarray(targets)
    if hasattr(dataset, 'train_labels'):
        targets = dataset.train_labels
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        return np.asarray(targets)
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels)
    raise AttributeError('Cannot infer labels/targets from dataset')


def dirichlet_partition(dataset, num_users, alpha, min_size=20, max_retries=100):
    # 使用 Dirichlet 分布做 Non-IID 划分。
    # 核心思想：对“每个类别”分别生成一个长度为 num_users 的比例向量 p ~ Dir([alpha]*K)，
    # 然后把该类别的样本按照 p 分配到各个客户端。
    # - alpha 越小：p 越稀疏，客户端更容易只拿到少数类别（更异构）
    # - alpha 越大（如 100）：p 越接近均匀分布（统计意义上近似 IID）
    #
    # 鲁棒性：当 alpha 很小时，某些客户端可能分不到样本。
    # 这里通过“重采样 max_retries 次 + 最小样本数 min_size 约束”来避免空客户端/小客户端。
    targets = _get_targets_numpy(dataset)
    num_classes = int(np.max(targets)) + 1

    for _ in range(max_retries):
        dict_users = {i: [] for i in range(num_users)}

        for c in range(num_classes):
            idx_c = np.where(targets == c)[0]
            np.random.shuffle(idx_c)

            # 对类别 c 生成 K 个客户端的分配比例（和为 1）
            proportions = np.random.dirichlet([alpha] * num_users)
            # 将该类别的样本数 len(idx_c) 按比例采样为整数计数（每个客户端拿 cnt 个）
            counts = np.random.multinomial(len(idx_c), proportions)

            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt

        sizes = [len(v) for v in dict_users.values()]
        if min(sizes) >= min_size:
            # 返回格式必须是 {user_id: np.array(indices)}，以兼容现有 LocalUpdate/DatasetSplit
            dict_users = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users.items()}

            all_idxs = np.concatenate(list(dict_users.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
            if all_idxs.size != len(dataset):
                raise ValueError(
                    f"Dirichlet partition produced wrong total size: {all_idxs.size} vs {len(dataset)}"
                )
            if np.any(all_idxs < 0) or np.any(all_idxs >= len(dataset)):
                raise ValueError("Dirichlet partition produced out-of-range indices")
            if np.unique(all_idxs).size != all_idxs.size:
                raise ValueError("Dirichlet partition produced duplicate indices across users")

            return dict_users

    dict_users = {i: [] for i in range(num_users)}
    all_idxs = np.random.permutation(len(dataset)).tolist()
    for i, idx in enumerate(all_idxs):
        dict_users[i % num_users].append(idx)

    # 提示（用于后续验证每个客户端的类别分布，可画 heatmap）：
    # targets = _get_targets_numpy(dataset)
    # for client_id, idxs in dict_users.items():
    #     hist = np.bincount(targets[np.asarray(idxs, dtype=np.int64)], minlength=num_classes)
    #     # hist 即该客户端各类别样本数
    return {k: np.asarray(v, dtype=np.int64) for k, v in dict_users.items()}
