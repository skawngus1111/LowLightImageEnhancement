import torch

import numpy as np
from tqdm import tqdm

from LLIEExperiment._base import BaseExperiment
from utils.calculate_metrics import compute_measure
from utils.save_functions import save_model
from utils.load_functions import load_model

class LowLightImageEnhancement(BaseExperiment):
    def __init__(self, args):
        super(LowLightImageEnhancement, self).__init__(args)

        self.eval_metrics = {
            "psnr": [],
            "ssim": [],
            "lpips": []
        }
        self.psnr_max = 0

    def fit(self):
        if self.args.train:
            self.loss_list = []
            print("################ Train ################")
            for epoch in range(1, 10 + 1):
                self.train_epoch()
                self.scheduler.step()
                self.val_epoch(epoch)

        print("################ Inference ##############")
        self.inference()

    def train_epoch(self):
        self.model.train()
        for data_batch in tqdm(self.train_loader):
            output_batch = self.forward(data_batch)
            self.backward(output_batch['loss'])
            self.loss_list.append(output_batch['loss'].item())

    def val_epoch(self, epoch):
        self.model.eval()

        psnr_list, ssim_list, lpips = [], [], 0
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            for counter, data_batch in enumerate(tqdm(self.test_loader)):
                output_batch = self.forward(data_batch)
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(np.transpose(output_batch['prediction'].squeeze().cpu().detach().numpy(), (1, 2, 0)))
                # ax[1].imshow(np.transpose(data_batch['HQ_image'].squeeze().cpu().detach().numpy(), (1, 2, 0)))
                # plt.show()
                psnr, ssim = compute_measure(output_batch['prediction'], data_batch['HQ_image'])
                psnr_list.append(psnr)
                ssim_list.append(ssim)

        c_psnr = np.average(psnr_list)
        c_ssim = np.average(ssim_list)
        print("EPOCH {} | Average PSNR: {} | SSIM: {}".format(epoch, c_psnr, c_ssim))
        if c_psnr >= self.psnr_max:
            self.psnr_max = c_psnr
            save_model(self.args, self.model)
            print("SAVE BEST MODEL (PNSR {} | SSIM {}) IN EPOCH {}".format(c_psnr, c_ssim, epoch))

    def inference(self):
        self.model = load_model(self.args, self.model)
        self.model.eval()

        psnr_list, ssim_list, lpips = [], [], 0
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            for counter, data_batch in enumerate(tqdm(self.test_loader)):
                output_batch = self.forward(data_batch)
                psnr, ssim = compute_measure(output_batch['prediction'], data_batch['HQ_image'])
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        print("Average PSNR: {} | SSIM: {}".format(np.average(psnr_list), np.average(ssim_list)))