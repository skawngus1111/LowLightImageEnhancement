from contextlib import nullcontext

import torch

import numpy as np

from dataset import dataloader_generator
from utils.get_functions import get_device
from utils.scheduler import GradualWarmupScheduler, CosineAnnealingRestartCyclicLR
from model_list import low_light_image_enhancement_model

class BaseExperiment(object):
    def __init__(self, args):
        super(BaseExperiment, self).__init__()

        self.args = args
        self.args.device = get_device()
        self.scaler = torch.cuda.amp.GradScaler()
        self.train_loader, self.valid_loader, self.test_loader = dataloader_generator(args)

        self.model = low_light_image_enhancement_model(self.args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=self.optimizer, periods=[(self.final_epoch // 4) - 3, (self.final_epoch * 3) // 4], restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
        # self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_step)
        self.scheduler = adjust_learning_rate(
            self.optimizer,
            self.args.final_epoch,
            len(self.train_loader),
            self.args.lr
        )

    def forward(self, data_batch):
        data_batch = self.cpu_to_gpu(data_batch)
        ctx = torch.cuda.amp.autocast() if self.args.amp else nullcontext()
        with ctx: return self.model(data_batch, self.lpips_fn)

    def backward(self, loss):
        self.optimizer.zero_grad()

        if self.args.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # self.scheduler.step()

    def cpu_to_gpu(self, data):
        dev = self.args.device if isinstance(self.args.device, torch.device) else torch.device(str(self.args.device))
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(dev, non_blocking=True)
        return data

def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def adjust_learning_rate(optimizer, epochs: int, train_loader_len: int, learning_rate: float):
    """
    Cosine annealing 스케줄러 래퍼.
    - optimizer: torch.optim.Optimizer
    - epochs: 총 epoch 수
    - train_loader_len: len(train_loader)
    - learning_rate: initial lr
    """
    total_steps = epochs * train_loader_len

    def lr_lambda(step: int) -> float:
        # lr_lambda는 lr multiplicative factor를 반환해야 함.
        return get_lr(step, total_steps, lr_max=1.0, lr_min=1e-6 / learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler