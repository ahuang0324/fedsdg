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


def dirichlet_partition_train_test(train_dataset, test_dataset, num_users, alpha, 
                                    train_ratio=0.8, min_size=20, max_retries=100):
    """
    同时划分训练集和测试集，确保每个客户端的训练集和测试集具有相同的标签分布。
    
    双重评估机制核心函数：
    1. 对训练集使用 Dirichlet 分布划分，生成 dict_users_train
    2. 对测试集按照相同的类别比例划分，生成 dict_users_test
    3. 保证 dict_users_test[client_id] 的标签分布与 dict_users_train[client_id] 一致
    
    参数:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        num_users: 客户端数量
        alpha: Dirichlet 分布参数（越小越异构）
        train_ratio: 训练集占比（用于从训练集中划分本地验证集，此处不使用）
        min_size: 每个客户端最小样本数
        max_retries: 最大重试次数
    
    返回:
        dict_users_train: {client_id: np.array(train_indices)}
        dict_users_test: {client_id: np.array(test_indices)}
    """
    train_targets = _get_targets_numpy(train_dataset)
    test_targets = _get_targets_numpy(test_dataset)
    num_classes = int(np.max(train_targets)) + 1
    
    for _ in range(max_retries):
        dict_users_train = {i: [] for i in range(num_users)}
        dict_users_test = {i: [] for i in range(num_users)}
        
        # 记录每个客户端在每个类别上的比例（用于测试集划分）
        client_class_proportions = np.zeros((num_users, num_classes))
        
        # ==================== 第一步：划分训练集 ====================
        for c in range(num_classes):
            idx_c = np.where(train_targets == c)[0]
            np.random.shuffle(idx_c)
            
            # 对类别 c 生成 K 个客户端的分配比例
            proportions = np.random.dirichlet([alpha] * num_users)
            client_class_proportions[:, c] = proportions
            
            # 按比例分配样本
            counts = np.random.multinomial(len(idx_c), proportions)
            
            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users_train[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt
        
        # 检查训练集划分是否满足最小样本数约束
        train_sizes = [len(v) for v in dict_users_train.values()]
        if min(train_sizes) < min_size:
            continue  # 重试
        
        # ==================== 第二步：按相同比例划分测试集 ====================
        # 使用训练集划分时的相同比例来划分测试集
        for c in range(num_classes):
            idx_c = np.where(test_targets == c)[0]
            np.random.shuffle(idx_c)
            
            # 使用训练集划分时的相同比例
            proportions = client_class_proportions[:, c]
            
            # 归一化（防止数值误差）
            proportions = proportions / (proportions.sum() + 1e-10)
            
            # 按比例分配测试样本
            counts = np.random.multinomial(len(idx_c), proportions)
            
            start = 0
            for user_id, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                dict_users_test[user_id].extend(idx_c[start:start + cnt].tolist())
                start += cnt
        
        # ==================== 验证划分结果 ====================
        # 转换为 numpy 数组
        dict_users_train = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_train.items()}
        dict_users_test = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_test.items()}
        
        # 验证训练集
        all_train_idxs = np.concatenate(list(dict_users_train.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
        if all_train_idxs.size != len(train_dataset):
            raise ValueError(f"Train partition size mismatch: {all_train_idxs.size} vs {len(train_dataset)}")
        if np.unique(all_train_idxs).size != all_train_idxs.size:
            raise ValueError("Train partition has duplicate indices")
        
        # 验证测试集
        all_test_idxs = np.concatenate(list(dict_users_test.values())) if num_users > 0 else np.asarray([], dtype=np.int64)
        if all_test_idxs.size != len(test_dataset):
            raise ValueError(f"Test partition size mismatch: {all_test_idxs.size} vs {len(test_dataset)}")
        if np.unique(all_test_idxs).size != all_test_idxs.size:
            raise ValueError("Test partition has duplicate indices")
        
        return dict_users_train, dict_users_test
    
    # ==================== 回退策略：均匀划分 ====================
    print("[Warning] Dirichlet partition failed after max retries, falling back to uniform partition")
    
    dict_users_train = {i: [] for i in range(num_users)}
    dict_users_test = {i: [] for i in range(num_users)}
    
    # 均匀划分训练集
    all_train_idxs = np.random.permutation(len(train_dataset)).tolist()
    for i, idx in enumerate(all_train_idxs):
        dict_users_train[i % num_users].append(idx)
    
    # 均匀划分测试集
    all_test_idxs = np.random.permutation(len(test_dataset)).tolist()
    for i, idx in enumerate(all_test_idxs):
        dict_users_test[i % num_users].append(idx)
    
    dict_users_train = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_train.items()}
    dict_users_test = {k: np.asarray(v, dtype=np.int64) for k, v in dict_users_test.items()}
    
    return dict_users_train, dict_users_test
