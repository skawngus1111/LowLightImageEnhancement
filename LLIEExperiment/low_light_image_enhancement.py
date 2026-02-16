import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import lpips
import numpy as np
from tqdm import tqdm

from LLIEExperiment._base import BaseExperiment
from utils.calculate_metrics import compute_measure
from utils.save_functions import save_model, save_predictions, save_loss_graph
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
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.args.device).eval()

    def fit(self):
        if self.args.train:
            # self.model.load_state_dict(torch.load('Epoch99.pth'))
            self.train_loss_dict = {
                'total_loss': [],
                'recon_loss': [],
                'ssim_loss': [],
                'lpips_loss': [],
                'color_loss': [],
                'edge_loss': []
            }
            self.val_loss_dict = {
                'total_loss': [],
                'recon_loss': [],
                'ssim_loss': [],
                'lpips_loss': [],
                'color_loss': [],
                'edge_loss': []
            }

            print("################ Train ################")
            for epoch in range(1, self.args.final_epoch + 1):
                self.psnr_list, self.ssim_list, self.lpips_list, self.c_loss_list = [], [], [], []
                self.args.current_epoch = epoch
                self.train_epoch(epoch)
                self.scheduler.step()
                # if epoch % 10 == 0:
                    # save_model(self.args, self.model)
                self.val_epoch(epoch)

        save_loss_graph(self.args, self.train_loss_dict, self.val_loss_dict)
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

        train_result_dict = {
            'total_loss': 0,
            'recon_loss': 0,
            'ssim_loss': 0,
            'lpips_loss': 0,
            'color_loss': 0,
            'edge_loss': 0
        }

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}/{self.args.final_epoch}", leave=True)

        for step, data_batch in enumerate(pbar, 1):
            output_batch = self.forward(data_batch)
            self.backward(output_batch['loss'])

            loss_val = float(output_batch['loss'].detach().item()) if torch.is_tensor(output_batch['loss']) else float(
                output_batch['loss'])
            ld = output_batch.get('loss_dict', {})

            train_result_dict['total_loss'] += ld.get('total_loss', loss_val)
            train_result_dict['recon_loss'] += ld.get('recon', 0.0)
            train_result_dict['ssim_loss'] += ld.get('ssim', 0.0)
            train_result_dict['lpips_loss'] += ld.get('lpips', 0.0)
            train_result_dict['color_loss'] += ld.get('color', 0.0)
            train_result_dict['edge_loss'] += ld.get('edge', 0.0)

            pbar.set_postfix({
                "loss": f"{loss_val:.6f}",
                "recon": f"{ld.get('recon', 0.0):.6f}",
                "ssim": f"{ld.get('ssim', 0.0):.6f}",
                "lpips": f"{ld.get('lpips', 0.0):.6f}",
                "color": f"{ld.get('color', 0.0):.6f}",
                "edge": f"{ld.get('edge', 0.0):.6f}",
            })

        # ---- epoch 평균 계산 ----
        n = len(self.train_loader)
        avg_total = train_result_dict['total_loss'] / n
        avg_recon = train_result_dict['recon_loss'] / n
        avg_ssim = train_result_dict['ssim_loss'] / n
        avg_lpips = train_result_dict['lpips_loss'] / n
        avg_color = train_result_dict['color_loss'] / n
        avg_edge = train_result_dict['edge_loss'] / n

        # dict에도 평균으로 저장(기존 코드와 호환)
        train_result_dict['total_loss'] = avg_total
        train_result_dict['recon_loss'] = avg_recon
        train_result_dict['ssim_loss'] = avg_ssim
        train_result_dict['lpips_loss'] = avg_lpips
        train_result_dict['color_loss'] = avg_color
        train_result_dict['edge_loss'] = avg_edge

        # ---- epoch 종료 로그 출력 (tqdm와 깔끔하게 공존) ----
        tqdm.write(
            f"[Epoch {epoch:03d}] "
            f"avg_total={avg_total:.6f} | avg_recon={avg_recon:.6f} | "
            f"avg_ssim={avg_ssim:.6f} | avg_lpips={avg_lpips:.6f} | "
            f"avg_color={avg_color:.6f} | avg_edge={avg_edge:.6f}"
        )

        # (선택) 마지막 progress bar postfix를 평균으로 덮어쓰기
        pbar.set_postfix({
            "avg_total": f"{avg_total:.6f}",
            "avg_recon": f"{avg_recon:.6f}",
            "avg_ssim": f"{avg_ssim:.6f}",
            "avg_lpips": f"{avg_lpips:.6f}",
            "avg_color": f"{avg_color:.6f}",
            "avg_edge": f"{avg_edge:.6f}",
        })

        self.train_loss_dict['total_loss'].append(avg_total)
        self.train_loss_dict['recon_loss'].append(avg_recon)
        self.train_loss_dict['ssim_loss'].append(avg_ssim)
        self.train_loss_dict['lpips_loss'].append(avg_lpips)
        self.train_loss_dict['color_loss'].append(avg_color)
        self.train_loss_dict['edge_loss'].append(avg_edge)

    # def val_epoch(self, epoch):
    #     self.model.eval()
    #     psnr_list, ssim_list, lpips = [], [], 0
    #     ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    #     with ctx():
    #         for counter, data_batch in enumerate(tqdm(self.valid_loader)):
    #             output_batch = self.forward(data_batch)
    #             psnr, ssim = compute_measure(output_batch['prediction'], data_batch['HQ_image'])
    #             psnr_list.append(psnr)
    #             ssim_list.append(ssim)
    #
    #             save_predictions(self.args, output_batch['prediction'], counter, validation=True, psnr=psnr, ssim=ssim)
    #
    #     c_psnr = np.average(psnr_list)
    #     c_ssim = np.average(ssim_list)
    #     print("EPOCH {} | Average PSNR: {} | SSIM: {}".format(epoch, c_psnr, c_ssim))


    def val_epoch(self, epoch):
        self.model.eval()

        psnr_list, ssim_list, lpips_list = [], [], []

        val_result = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "ssim_loss": 0.0,
            "lpips_loss": 0.0,
            "color_loss": 0.0,
            "edge_loss": 0.0
        }
        has_loss_dict = False  # loss_dict가 실제로 들어오는지 체크

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            pbar = tqdm(self.valid_loader, desc=f"Val {epoch}/{self.args.final_epoch}", leave=True)

            for counter, data_batch in enumerate(pbar, 1):
                output_batch = self.forward(data_batch)

                # ---- metrics ----
                psnr, ssim, lp = compute_measure(output_batch['prediction'], data_batch['HQ_image'], lpips_fn=self.lpips_fn)
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                lpips_list.append(float(lp))

                # ---- loss ----
                ld = output_batch.get("loss_dict", {})
                if isinstance(ld, dict) and len(ld) > 0:
                    has_loss_dict = True
                    val_result["total_loss"] += float(ld.get("total", 0.0))
                    val_result["recon_loss"] += float(ld.get("recon", 0.0))
                    val_result["ssim_loss"] += float(ld.get("ssim", 0.0))
                    val_result["lpips_loss"] += float(ld.get("lpips", 0.0))
                    val_result["color_loss"] += float(ld.get("color", 0.0))
                    val_result["edge_loss"] += float(ld.get("edge", 0.0))

                # ---- save image ----
                if epoch % 10 == 0:
                    save_predictions(
                        self.args,
                        output_batch['prediction'],
                        counter - 1,
                        validation=True,
                        psnr=psnr,
                        ssim=ssim
                    )

                # ---- running averages ----
                avg_psnr = float(np.mean(psnr_list))
                avg_ssim = float(np.mean(ssim_list))
                avg_lpips = float(np.mean(lpips_list))

                # ---- current loss values (if available) ----
                cur_total = float(ld.get("total_loss", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0
                cur_recon = float(ld.get("recon", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0
                cur_ssim_l = float(ld.get("ssim", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0
                cur_lpips_loss = float(ld.get("lpips", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0
                cur_color_loss = float(ld.get("color", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0
                cur_edge_loss = float(ld.get("edge", 0.0)) if has_loss_dict and len(ld) > 0 else 0.0

                postfix = {
                    "P": f"{psnr:.3f}",
                    "S": f"{ssim:.4f}",
                    "LPI": f"{lp:.4f}",
                    "mP": f"{avg_psnr:.3f}",
                    "mS": f"{avg_ssim:.4f}",
                    "mLPI": f"{avg_lpips:.4f}",
                }

                if has_loss_dict and len(ld) > 0:
                    postfix.update({
                        "total_loss": f"{cur_total:.4f}",
                        "rec": f"{cur_recon:.4f}",
                        "ssimL": f"{cur_ssim_l:.4f}",
                        "lpips_loss": f"{cur_lpips_loss:.4f}",
                        "color_loss": f"{cur_color_loss:.4f}",
                        "edge_loss": f"{cur_edge_loss:.4f}",
                    })

                pbar.set_postfix(postfix)

        # ---- epoch averages ----
        n = len(self.valid_loader)
        c_psnr = float(np.mean(psnr_list)) if len(psnr_list) else 0.0
        c_ssim = float(np.mean(ssim_list)) if len(ssim_list) else 0.0
        c_lpips = float(np.mean(lpips_list)) if len(lpips_list) else 0.0

        loss_str = ""
        if has_loss_dict and n > 0:
            avg_total = val_result["total_loss"] / n
            avg_recon = val_result["recon_loss"] / n
            avg_ssim_l = val_result["ssim_loss"] / n
            avg_lpips_loss = val_result["lpips_loss"] / n
            avg_color_loss = val_result["color_loss"] / n
            avg_edge_loss = val_result["edge_loss"] / n
            loss_str = (f" | avg_loss={avg_total:.6f} "
                        f"(recon={avg_recon:.6f}, ssim={avg_ssim_l:.6f}, lpips_loss={avg_lpips_loss:.6f}, color_loss={avg_color_loss:.6f}, edge_loss={avg_edge_loss:.6f})")

        print(f"EPOCH {epoch} | PSNR={c_psnr:.4f} | SSIM={c_ssim:.6f} | LPIPS={c_lpips:.6f}{loss_str}")

        self.val_loss_dict['total_loss'].append(val_result["total_loss"] / n)
        self.val_loss_dict['recon_loss'].append(val_result["recon_loss"] / n)
        self.val_loss_dict['ssim_loss'].append(val_result["ssim_loss"] / n)
        self.val_loss_dict['lpips_loss'].append(val_result["lpips_loss"] / n)
        self.val_loss_dict['color_loss'].append(val_result["color_loss"] / n)
        self.val_loss_dict['edge_loss'].append(val_result["edge_loss"] / n)

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