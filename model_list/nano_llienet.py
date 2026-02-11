import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rgb_to_luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def GN(ch, max_groups=8, eps=1e-5, affine=True):
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch, eps=eps, affine=affine)


# ============================================================
# Guided Filter (Fast, O(n), edge-preserving de-posterization)
# ============================================================
def box_filter(x, r):
    """Box filter via avg_pool2d. O(n) regardless of radius."""
    return F.avg_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)


def guided_filter(x, guide, r=3, eps=0.02):
    """
    Fast Guided Filter for de-posterization.

    원리: guide 이미지의 edge 구조를 보존하면서 smoothing.
    - guide의 edge가 있는 곳 → 거의 안 바꿈 (진짜 edge)
    - guide의 edge가 없는 곳 → 평균으로 부드럽게 (posterization 제거)

    Parameters:
        x:     입력 (gamma-corrected, posterized) [B, C, H, W]
        guide: guide 이미지 (luma) [B, 1, H, W]
        r:     filter radius (3이면 7×7 window)
        eps:   regularization (클수록 더 smooth, 0.01~0.05)

    Returns:
        filtered: de-posterized [B, C, H, W]
    """
    C = x.shape[1]

    # Guide statistics (1-channel)
    mean_I = box_filter(guide, r)
    mean_II = box_filter(guide * guide, r)
    var_I = mean_II - mean_I * mean_I  # guide의 local variance

    # Per-channel filtering
    mean_p = box_filter(x, r)
    mean_Ip = box_filter(guide.expand_as(x) * x, r)
    cov_Ip = mean_Ip - mean_I * mean_p  # guide와 input의 covariance

    # Linear coefficients: output = a * guide + b
    # var_I가 큰 곳(=진짜 edge) → a가 큼 → guide를 따라감 (edge 보존)
    # var_I가 작은 곳(=posterization) → a≈0, b≈mean → 평균으로 smooth
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Average coefficients over window for stability
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    return mean_a * guide.expand_as(x) + mean_b


# ============= Ghost Module =============
class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2, dw_size=3, stride=1, relu=True, gn_groups=8):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, 1, stride, 0, bias=False),
            GN(init_channels, max_groups=gn_groups),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            GN(new_channels, max_groups=gn_groups),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


# ============= Spatial Attention =============
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


# ============= Channel Attention =============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


# ============= Enhanced Ghost Block =============
class EnhancedGhostBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, gn_groups=8):
        super().__init__()
        self.ghost = GhostModule(in_ch, out_ch, stride=stride, gn_groups=gn_groups)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                GN(out_ch, max_groups=gn_groups),
            )

    def forward(self, x):
        out = self.ghost(x)
        out = self.ca(out)
        out = self.sa(out)
        return out + self.shortcut(x)


# ============= Multi-Scale Fusion =============
class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_low, x_high):
        if x_low.shape[2:] != x_high.shape[2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_low, x_high], dim=1)
        x_cat = self.conv1x1(x_cat)
        x3 = self.conv3x3(x_cat)
        x5 = self.conv5x5(x_cat)
        out = self.fusion(torch.cat([x3, x5], dim=1))
        return self.act(out)


# ============= Main Network =============
class NanoLLE(nn.Module):
    """
    v4: Multi-Gamma + Guided Filter + Gain-map + Additive Bias

    Pipeline:
      1. Input x → gamma(x, 1/g) for g in [2,3,4,5] → 각각 guided filter
      2. cat[x, gf(x^1/2), gf(x^1/3), gf(x^1/4), gf(x^1/5)] → 15ch
      3. Encoder-Decoder → features
      4. Gain-map (곱셈, 구조 보존) + Bias (덧셈, 어두운 영역) + Residual
    """

    def __init__(self, base_channels=24, depths=(2, 2, 3), gn_groups=8,
                 gamma_values=(2.0, 3.0, 4.0, 5.0),
                 gf_radius=3, gf_eps=0.02):
        super().__init__()
        C = base_channels
        d0, d1, db = depths
        self.gamma_values = gamma_values
        self.gf_radius = gf_radius
        self.gf_eps = gf_eps
        n_gamma = len(gamma_values)

        # Multi-gamma input: original(3) + K gamma versions(3*K) = 15ch
        in_ch = 3 * (1 + n_gamma)

        # Stem: 15ch → C
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
            GN(C, max_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc0 = nn.Sequential(*[EnhancedGhostBlock(C, C, gn_groups=gn_groups) for _ in range(d0)])
        self.down1 = EnhancedGhostBlock(C, 2 * C, stride=2, gn_groups=gn_groups)
        self.enc1 = nn.Sequential(*[EnhancedGhostBlock(2 * C, 2 * C, gn_groups=gn_groups) for _ in range(d1)])
        self.down2 = EnhancedGhostBlock(2 * C, 4 * C, stride=2, gn_groups=gn_groups)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[EnhancedGhostBlock(4 * C, 4 * C, gn_groups=gn_groups) for _ in range(db)])

        # Decoder
        self.up2_conv = EnhancedGhostBlock(4 * C, 2 * C, gn_groups=gn_groups)
        self.fuse1 = MultiScaleFusion(2 * C)
        self.up1_conv = EnhancedGhostBlock(2 * C, C, gn_groups=gn_groups)
        self.fuse0 = MultiScaleFusion(C)

        # ===== Output Heads =====
        # (1) Gain Map: 곱셈 (구조/색 보존)
        self.gain_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 3, 1),
        )

        # (2) Additive Bias: 덧셈 (x≈0 영역 직접 밝히기)
        self.bias_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 3, 1),
        )

        # (3) Refinement: 미세 보정
        self.refine = nn.Sequential(
            nn.Conv2d(C + 3, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 3, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_multi_gamma_input(self, x):
        """
        Multi-gamma + debanding.

        x: [B, 3, H, W] in [0,1]
        return: [B, 3*(1+len(gamma_values)), H, W]
        """
        inputs = [x]

        # (1) pre: de-quantization / dithering (8-bit banding 깨기)
        # scale은 상황에 따라 1/255 ~ 2/255 정도부터 시작 추천
        if getattr(self, "dither_strength", 0.0) > 0:
            x = (x + (torch.rand_like(x) - 0.5) * self.dither_strength).clamp(0.0, 1.0)

        x_safe = x.clamp(min=1e-6)

        # (2) guide는 gamma_img가 아니라 "원본 기반" (밴딩 엣지 보존 방지)
        # 원본이 너무 어두우면 log-luma가 더 안정적일 때가 많음
        guide = rgb_to_luma(x_safe)
        if getattr(self, "use_log_guide", True):
            guide = torch.log(guide + 1e-6)

        for g in self.gamma_values:
            gamma_img = torch.exp(torch.log(x_safe) / g)

            # (3) post: guided debanding (p=gamma_img, I=guide)
            gamma_filtered = guided_filter(
                gamma_img, guide,
                r=self.gf_radius, eps=self.gf_eps
            ).clamp(0.0, 1.0)

            inputs.append(gamma_filtered)

        return torch.cat(inputs, dim=1)

    def _plot_multi_x(self, multi_x, step=0, out_dir="debug_multi_x"):
        """
        multi_x: [B, 3*(1+len(gamma_values)), H, W] in [0,1]
        배치 0번 샘플만, 3채널씩 끊어서 저장.
        """
        import matplotlib.pyplot as plt
        # os.makedirs(out_dir, exist_ok=True)

        mx = multi_x[0].detach().float().cpu().clamp(0, 1)  # [C,H,W]
        n_views = mx.shape[0] // 3

        for i in range(n_views):
            img = mx[i * 3:(i + 1) * 3].permute(1, 2, 0).numpy()  # [H,W,3]
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"view {i}")
            plt.show()

    def forward(self, data_batch):
        x = data_batch["LQ_image"]
        gt = data_batch.get("HQ_image", None)

        # ===== Multi-gamma + de-posterization =====
        multi_x = self._build_multi_gamma_input(x)  # [B, 15, H, W]
        # print(multi_x.shape)
        # self._plot_multi_x(multi_x)


        # ===== Encoder =====
        f = self.stem(multi_x)
        s0 = self.enc0(f)

        f = self.down1(s0)
        s1 = self.enc1(f)

        f = self.down2(s1)
        f = self.bottleneck(f)

        # ===== Decoder =====
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up2_conv(f)
        f = self.fuse1(f, s1)

        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up1_conv(f)
        f = self.fuse0(f, s0)

        # =====================================================
        # Gain × input + Bias + Residual
        # =====================================================
        raw_gain = self.gain_head(f)
        gain_map = 1.0 + F.softplus(raw_gain)  # >= 1.0

        bias = torch.sigmoid(self.bias_head(f)) * 0.5  # [0, 0.5]

        enhanced = (x * gain_map + bias).clamp(0.0, 1.0)

        combined = torch.cat([f, enhanced], dim=1)
        residual = self.refine(combined)

        pred = (enhanced + residual).clamp(0.0, 1.0)

        # ===== Loss =====
        if gt is not None:
            loss, loss_dict = self._calculate_loss(data_batch, pred, gt, gain_map)
        else:
            loss, loss_dict = None, {}

        return {
            "prediction": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "illum_map": gain_map[:, 0:1],
            "color_correction": gain_map,
        }

    # ============================================================
    # SSIM Loss
    # ============================================================
    def _ssim_loss(self, pred, gt):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        win = 11
        sigma = 1.5

        coords = torch.arange(win, dtype=pred.dtype, device=pred.device) - win // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        ch = pred.shape[1]
        window = window.expand(ch, -1, -1, -1).contiguous()
        pad = win // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
        mu2 = F.conv2d(gt, window, padding=pad, groups=ch)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(gt ** 2, window, padding=pad, groups=ch) - mu2_sq
        sigma12 = F.conv2d(pred * gt, window, padding=pad, groups=ch) - mu1_mu2

        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()

    # ============================================================
    # Loss
    # ============================================================
    def _calculate_loss(self, data_batch, pred, gt, gain_map):
        L_in = rgb_to_luma(data_batch["LQ_image"]).clamp(0, 1)
        w = dark_weight(L_in)

        # (1) Dark-weighted L1
        recon_l1 = (w * (pred - gt).abs()).mean()

        # (2) Dark-weighted gradient loss
        Lp = rgb_to_luma(pred)
        Lg = rgb_to_luma(gt)
        gx_p, gy_p = grad_xy(Lp)
        gx_g, gy_g = grad_xy(Lg)
        w_x = w[:, :, :, 1:]
        w_y = w[:, :, 1:, :]
        grad_loss = (w_x * (gx_p - gx_g).abs()).mean() + \
                    (w_y * (gy_p - gy_g).abs()).mean()

        # (3) Color loss
        color_loss = F.l1_loss(
            pred.mean(dim=[2, 3], keepdim=True),
            gt.mean(dim=[2, 3], keepdim=True)
        )

        # (4) SSIM loss
        ssim_loss = self._ssim_loss(pred, gt)

        # (5) Gain smoothness
        gx = gain_map[:, :, :, 1:] - gain_map[:, :, :, :-1]
        gy = gain_map[:, :, 1:, :] - gain_map[:, :, :-1, :]
        smooth_loss = gx.abs().mean() + gy.abs().mean()

        total_loss = (
                1.0 * recon_l1 +
                0.2 * grad_loss +
                0.2 * color_loss +
                0.3 * ssim_loss +
                0.02 * smooth_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "recon": recon_l1.item(),
            "grad": grad_loss.item(),
            "color": color_loss.item(),
            "ssim": ssim_loss.item(),
            "smooth": smooth_loss.item(),
        }
        return total_loss, loss_dict


# ============================================================
# Utility functions
# ============================================================
def dark_weight(L, tau=0.15, s=0.06, lam=2.0, p=2.0, wmin=0.5, wmax=5.0):
    m = torch.sigmoid((tau - L) / s)
    w = 1.0 + lam * (m ** p)
    w = torch.clamp(w, wmin, wmax)
    w = w / (w.mean(dim=[2, 3], keepdim=True) + 1e-6)
    return w


def grad_xy(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


if __name__ == "__main__":
    model = NanoLLE(base_channels=24, depths=(2, 2, 3), gn_groups=8)
    print(f"Model Size: {get_model_size(model):.3f} MB")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")

    dummy = {
        "LQ_image": torch.randn(1, 3, 256, 256).clamp(0, 1),
        "HQ_image": torch.randn(1, 3, 256, 256).clamp(0, 1),
    }
    with torch.no_grad():
        out = model(dummy)
    print(f"pred: {out['prediction'].shape}")
    print(f"loss: {out['loss'].item():.4f}")
    print(f"loss_dict: {out['loss_dict']}")