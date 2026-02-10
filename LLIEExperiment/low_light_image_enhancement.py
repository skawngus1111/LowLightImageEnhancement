import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from LLIEExperiment._base import BaseExperiment
from utils.calculate_metrics import compute_measure
from utils.save_functions import save_model, save_predictions
from utils.load_functions import load_model

class LowLightImageEnhancement(BaseExperiment):
    def __init__(self, args):
        super(LowLightImageEnhancement, self).__init__(args)

        self.size_rates = [0.75, 1, 1.25]

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
                # self.scheduler.step()
                # self.val_epoch(epoch)
                if epoch % 10 == 0:
                    # save_model(self.args, self.model)
                    self.inference_no_groundtruth()

        save_model(self.args, self.model)
        print("################ Inference ##############")
        # self.inference()
        self.inference_no_groundtruth()

    # def train_epoch(self, epoch):
    #     self.model.train()
    #     for data_batch in tqdm(self.train_loader):
    #         LQ_origina_image, HQ_origina_image = data_batch['LQ_image'], data_batch['HQ_image']
    #         for rate in self.size_rates:
    #             LQ_image, HQ_image = data_batch['LQ_image'], data_batch['HQ_image']
    #
    #             # ---- rescale ----
    #             trainsize = int(round(256 * rate / 32) * 32)
    #
    #             if rate != 1:
    #                 LQ_image = F.upsample(LQ_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
    #                 HQ_image = F.upsample(HQ_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
    #                 data_batch['LQ_image'] = LQ_image
    #                 data_batch['HQ_image'] = HQ_image
    #             else:
    #                 data_batch['LQ_image'] = LQ_origina_image
    #                 data_batch['HQ_image'] = HQ_origina_image
    #
    #             output_batch = self.forward(data_batch)
    #             self.backward(output_batch['loss'])
    #
    #             if rate == 1:
    #                 self.loss_list.append(output_batch['loss'].item())
    #                 self.c_loss_list.append(output_batch['loss'].item())
    #                 for prediction_, HQ_image_ in zip(output_batch['prediction'], data_batch['HQ_image']):
    #                     psnr, ssim = compute_measure(prediction_, HQ_image_)
    #                     self.psnr_list.append(psnr); self.ssim_list.append(ssim)
    #
    #     c_loss = np.average(self.c_loss_list)
    #     c_psnr = np.average(self.psnr_list)
    #     c_ssim = np.average(self.ssim_list)
    #     print("EPOCH {} | Loss: {} | Average PSNR: {} | SSIM: {}".format(epoch, c_loss, c_psnr, c_ssim))
    #     # if c_ssim >= self.ssim_max:
    #     #     self.ssim_max = c_ssim
    #     #     save_model(self.args, self.model)
    #     #     print("SAVE BEST MODEL (Loss {} | PNSR {} | SSIM {}) IN EPOCH {}".format(c_loss, c_psnr, c_ssim, epoch))

    def train_epoch(self, epoch):
        self.model.train()

        # 🔥 각 loss 구성 요소 추적용 딕셔너리 추가
        epoch_losses = {
            'total': [],
            'recon': [],
            'freq': [],
            'color': [],
            'smooth': []
        }

        for data_batch in tqdm(self.train_loader):
            LQ_original_image, HQ_original_image = data_batch['LQ_image'], data_batch['HQ_image']

            for rate in self.size_rates:
                LQ_image, HQ_image = data_batch['LQ_image'], data_batch['HQ_image']

                # ---- rescale ----
                trainsize = int(round(256 * rate / 32) * 32)

                if rate != 1:
                    LQ_image = F.upsample(LQ_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    HQ_image = F.upsample(HQ_image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    data_batch['LQ_image'] = LQ_image
                    data_batch['HQ_image'] = HQ_image
                else:
                    data_batch['LQ_image'] = LQ_original_image
                    data_batch['HQ_image'] = HQ_original_image

                output_batch = self.forward(data_batch)
                self.backward(output_batch['loss'])

                if rate == 1:
                    # 🔥 Loss 구성 요소 저장
                    loss_dict = output_batch['loss_dict']
                    for key in epoch_losses.keys():
                        if key in loss_dict:
                            epoch_losses[key].append(loss_dict[key])

                    # 기존 코드 (변경 없음)
                    self.loss_list.append(output_batch['loss'].item())
                    self.c_loss_list.append(output_batch['loss'].item())

                    for prediction_, HQ_image_ in zip(output_batch['prediction'], data_batch['HQ_image']):
                        psnr, ssim = compute_measure(prediction_, HQ_image_)
                        self.psnr_list.append(psnr)
                        self.ssim_list.append(ssim)

        # 🔥 평균 계산
        c_loss = np.average(self.c_loss_list)
        c_psnr = np.average(self.psnr_list)
        c_ssim = np.average(self.ssim_list)
        avg_losses = {key: np.average(values) for key, values in epoch_losses.items()}

        # 🔥 상세 출력
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}")
        print("=" * 70)
        print(f"Total Loss:  {c_loss:.4f}")
        print(f"  ├─ Recon:  {avg_losses['recon']:.4f}")
        print(f"  ├─ Freq:   {avg_losses['freq']:.4f}")
        print(f"  ├─ Color:  {avg_losses['color']:.4f}")
        print(f"  └─ Smooth: {avg_losses['smooth']:.4f}")
        print("-" * 70)
        print(f"PSNR: {c_psnr:.2f} dB | SSIM: {c_ssim:.4f}")
        print("=" * 70 + "\n")

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
        # self.model = load_model(self.args, self.model)
        self.model.eval()
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            for counter, data_batch in enumerate(tqdm(self.test_loader)):
                output_batch = self.forward(data_batch)
                save_predictions(self.args, output_batch['prediction'], counter)