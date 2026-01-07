#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import get_lora_state_dict


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs = list(idxs)
        np.random.shuffle(idxs)
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True,
                                 num_workers=4, pin_memory=True, prefetch_factor=2)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(1, int(len(idxs_val)/10)), shuffle=False,
                                 num_workers=2, pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(1, int(len(idxs_test)/10)), shuffle=False,
                                num_workers=2, pin_memory=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # FedLoRA: 仅优化可训练参数（LoRA 参数 + mlp_head）
        # FedAvg: 优化所有参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                global_step = (global_round * self.args.local_ep * len(self.trainloader)
                               + iter * len(self.trainloader) + batch_idx)
                self.logger.add_scalar('loss', loss.item(), global_step=global_step)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # FedLoRA/FedSDG: 仅返回 LoRA 参数（过滤私有参数和冻结权重）
        # FedAvg: 返回完整 state_dict
        if self.args.alg in ('fedlora', 'fedsdg'):
            return get_lora_state_dict(model), sum(epoch_loss) / len(epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, loader='train'):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if loader == 'train':
            dataloader = self.trainloader
        elif loader == 'val':
            dataloader = self.validloader
        elif loader == 'test':
            dataloader = self.testloader
        else:
            raise ValueError(f"Unknown loader: {loader}")

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    use_cuda = (args.gpu is not None) and (int(args.gpu) >= 0) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=4, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
