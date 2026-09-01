from model.nanoencoder.model import NanoAntiTFNet
from utils.warmup import GradualWarmupScheduler

from torch.utils.data import Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler

import torch


class WarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, max_lr, min_lr, power=2, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.power = power
        self.last_decay_lr = [group['lr'] for group in optimizer.param_groups]
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch, verbose=True)

    def get_lr(self):
        if self._step_count < self.warmup_iters:
            self.last_decay_lr = [base_lr + (self.max_lr - base_lr) * (self._step_count / self.warmup_iters) for base_lr in self.base_lrs]
        elif self._step_count < self.max_iters:
            decay_factor = (1 - (self._step_count - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power
            self.last_decay_lr = [self.max_lr * decay_factor + (1 - decay_factor) * base_lr for base_lr in self.base_lrs]
            if min(self.last_decay_lr) <= self.min_lr:
                self.last_decay_lr = [self.min_lr for _ in self.base_lrs]
        return self.last_decay_lr


def warmup(n_warmup_steps):
    def get_lr(step):
        return min((step + 1) / n_warmup_steps, 1.0)
    return get_lr


def model_selected(config):
    if config.name == 'nano':
        return NanoAntiTFNet(**config.model)
    else:
        raise NotImplementedError(f'Unknown model config name: {config.name}')


def optimizer_selected(optimizer, model):
    if optimizer.type == 'Adam':
        return Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay
        )
    elif optimizer.type == 'AdamW':
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay
        )
    else:
        raise NotImplementedError(f'Unknown optimizer type: {optimizer.type}')


def scheduler_selected(scheduler, optimizer):
    if scheduler.type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            factor=scheduler.factor,
            patience=scheduler.patience,
            min_lr=scheduler.min_lr
        )
    elif scheduler.type == 'cosine_annal':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler.T_max,
        )
    elif scheduler.type == 'warm_up':
        return WarmupPolyLR(
            optimizer,
            warmup_iters=scheduler.warmup_steps,
            max_iters=scheduler.max_steps,
            max_lr=scheduler.max_lr,
            min_lr=scheduler.min_lr
        )
    else:
        raise NotImplementedError(f'Unknown scheduler type: {scheduler.type}')
