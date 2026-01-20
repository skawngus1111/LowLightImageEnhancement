from contextlib import nullcontext

import torch

from dataset import dataloader_generator
from utils.get_functions import get_device
from utils.scheduler import GradualWarmupScheduler, CosineAnnealingRestartCyclicLR
from model_list import low_light_image_enhancement_model

class BaseExperiment(object):
    def __init__(self, args):
        super(BaseExperiment, self).__init__()

        self.args = args
        self.args.device = get_device()
        self.scaler = torch.amp.GradScaler()

        self.train_loader, self.test_loader = dataloader_generator(args)

        self.model = low_light_image_enhancement_model(self.args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=self.optimizer, periods=[(1000 // 4) - 3, (1000 * 3) // 4], restart_weights=[1, 1], eta_mins=[0.0002, 0.0000001])
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_step)

    def forward(self, data_batch):
        data_batch = self.cpu_to_gpu(data_batch)
        ctx = torch.cuda.amp.autocast() if self.args.amp else nullcontext()
        with ctx: return self.model(data_batch)

    def backward(self, loss):
        self.optimizer.zero_grad()

        if self.args.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def cpu_to_gpu(self, data):
        dev = self.args.device if isinstance(self.args.device, torch.device) else torch.device(str(self.args.device))
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(dev, non_blocking=True)
        return data