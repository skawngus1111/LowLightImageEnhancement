import torch

import numpy as np
from tqdm import tqdm

from LLIEExperiment._base import BaseExperiment
from utils.calculate_metrics import compute_measure
from utils.save_functions import save_model, save_predictions
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
        self.ssim_max = 0

    def fit(self):
        if self.args.train:
            self.loss_list = []
            print("################ Train ################")
            for epoch in range(1, self.final_epoch + 1):
                self.psnr_list, self.ssim_list, self.lpips_list, self.c_loss_list = [], [], [], []
                self.args.current_epoch = epoch
                self.train_epoch(epoch)
                self.scheduler.step()
                # self.val_epoch(epoch)
                if epoch % 10 == 0:
                    save_model(self.args, self.model)
                    self.inference_no_groundtruth()

        save_model(self.args, self.model)
        print("################ Inference ##############")
        # self.inference()
        self.inference_no_groundtruth()

    def train_epoch(self, epoch):
        self.model.train()
        for data_batch in tqdm(self.train_loader):
            output_batch = self.forward(data_batch)
            self.backward(output_batch['loss'])

            self.loss_list.append(output_batch['loss'].item())
            self.c_loss_list.append(output_batch['loss'].item())
            for prediction_, HQ_image_ in zip(output_batch['prediction'], data_batch['HQ_image']):
                psnr, ssim = compute_measure(prediction_, HQ_image_)
                self.psnr_list.append(psnr); self.ssim_list.append(ssim)

        c_loss = np.average(self.c_loss_list)
        c_psnr = np.average(self.psnr_list)
        c_ssim = np.average(self.ssim_list)
        print("EPOCH {} | Loss: {} | Average PSNR: {} | SSIM: {}".format(epoch, c_loss, c_psnr, c_ssim))
        # if c_ssim >= self.ssim_max:
        #     self.ssim_max = c_ssim
        #     save_model(self.args, self.model)
        #     print("SAVE BEST MODEL (Loss {} | PNSR {} | SSIM {}) IN EPOCH {}".format(c_loss, c_psnr, c_ssim, epoch))

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
        # if c_psnr >= self.psnr_max:
        #     self.psnr_max = c_psnr
        #     save_model(self.args, self.model)
        #     print("SAVE BEST MODEL (PNSR {} | SSIM {}) IN EPOCH {}".format(c_psnr, c_ssim, epoch))

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

    def inference_no_groundtruth(self):
        self.model = load_model(self.args, self.model)
        self.model.eval()
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            for counter, data_batch in enumerate(tqdm(self.test_loader)):
                output_batch = self.forward(data_batch)
                save_predictions(self.args, output_batch['prediction'], counter)