import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rgb_to_luma(x):
    return 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]

# ============= Ghost Module (효율적인 feature 생성) =============
class GhostModule(nn.Module):
    """
    Ghost Module: 적은 연산으로 더 많은 feature map 생성
    원리: 일부는 Conv로 생성, 나머지는 cheap operation(depthwise)으로 생성
    """

    def __init__(self, in_ch, out_ch, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        # Cheap operation (ghost features)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


# ============= Spatial Attention (공간적 중요 영역 강조) =============
class SpatialAttention(nn.Module):
    """
    밝기가 매우 낮은 영역과 세부 디테일이 있는 영역을 선택적으로 강조
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise max와 mean pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


# ============= Channel Attention (중요 feature channel 강조) =============
class ChannelAttention(nn.Module):
    """
    어떤 feature가 저조도 복원에 중요한지 학습
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# ============= Enhanced Ghost Block (Ghost + Attention) =============
class EnhancedGhostBlock(nn.Module):
    """
    Ghost Conv + Dual Attention으로 효율성과 성능 동시 확보
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.ghost = GhostModule(in_ch, out_ch, stride=stride)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.ghost(x)
        out = self.ca(out)
        out = self.sa(out)
        return out + self.shortcut(x)


# ============= Multi-Scale Fusion =============
class MultiScaleFusion(nn.Module):
    """
    다양한 스케일의 feature를 융합하여 서로 다른 조명 조건 대응
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_low, x_high):
        # Concatenate different resolution features
        if x_low.shape[2:] != x_high.shape[2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[2:],
                                  mode='bilinear', align_corners=False)

        x_cat = torch.cat([x_low, x_high], dim=1)
        x_cat = self.conv1x1(x_cat)

        # Multi-scale processing
        x3 = self.conv3x3(x_cat)
        x5 = self.conv5x5(x_cat)

        out = self.fusion(torch.cat([x3, x5], dim=1))
        return self.act(out)


# ============= Color Restoration Module =============
class ColorRestoration(nn.Module):
    """
    저조도에서 손실된 색상 정보 복원
    """

    def __init__(self, channels):
        super().__init__()
        # RGB 채널별 처리
        self.rgb_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.feature_conv = nn.Conv2d(channels, channels, 1)
        self.color_map = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x, features):
        rgb_feat = self.rgb_conv(x)
        combined = rgb_feat + self.feature_conv(features)
        color_correction = self.color_map(combined)
        return color_correction


# ============= Main Network =============
class NanoLLE(nn.Module):
    """
    개선된 Nano LLE Network

    주요 개선사항:
    1. Ghost Module로 효율성 극대화
    2. Dual Attention으로 중요 영역/채널 강조
    3. Multi-scale fusion으로 다양한 조명 대응
    4. Color restoration module로 색상 복원
    5. Advanced residual learning
    """

    def __init__(self, base_channels=24, depths=(2, 2, 3)):
        super().__init__()
        C = base_channels
        d0, d1, db = depths

        # ===== Encoder =====
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # Stage 0: Full resolution (H x W)
        self.enc0 = nn.Sequential(
            *[EnhancedGhostBlock(C, C) for _ in range(d0)]
        )

        # Downsample to H/2 x W/2
        self.down1 = EnhancedGhostBlock(C, 2 * C, stride=2)

        # Stage 1: Half resolution
        self.enc1 = nn.Sequential(
            *[EnhancedGhostBlock(2 * C, 2 * C) for _ in range(d1)]
        )

        # Downsample to H/4 x W/4
        self.down2 = EnhancedGhostBlock(2 * C, 4 * C, stride=2)

        # ===== Bottleneck =====
        self.bottleneck = nn.Sequential(
            *[EnhancedGhostBlock(4 * C, 4 * C) for _ in range(db)]
        )

        # ===== Decoder with Multi-scale Fusion =====
        # Upsample to H/2 x W/2
        self.up2_conv = EnhancedGhostBlock(4 * C, 2 * C)
        self.fuse1 = MultiScaleFusion(2 * C)

        # Upsample to H x W
        self.up1_conv = EnhancedGhostBlock(2 * C, C)
        self.fuse0 = MultiScaleFusion(C)

        # ===== Output Heads =====
        # Illumination map (밝기 조정)
        self.illum_head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, 1, 1),
            nn.Sigmoid()
        )

        # Color restoration
        self.color_restore = ColorRestoration(C)

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(C + 3, C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 3, 1)
        )

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data_batch):
        x = data_batch['LQ_image']
        gt = data_batch.get('HQ_image', None)

        # ===== Encoder =====
        f = self.stem(x)
        s0 = self.enc0(f)

        f = self.down1(s0)
        s1 = self.enc1(f)

        f = self.down2(s1)
        f = self.bottleneck(f)

        # ===== Decoder =====
        # H/4 -> H/2
        f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.up2_conv(f)
        f = self.fuse1(f, s1)

        # H/2 -> H
        f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.up1_conv(f)
        f = self.fuse0(f, s0)

        # ===== Multi-branch Prediction =====
        # Branch 1: Illumination adjustment
        # illum_map = self.illum_head(f)
        # enhanced_illum = x * illum_map + x  # Adaptive brightening

        illum_map = self.illum_head(f)
        L = rgb_to_luma(x).clamp(1e-4, 1.0)
        R = x / (L + 1e-4)
        R = R.clamp(0.0, 3.0)  # clamp(0,1) 제거!

        # Headroom 기반
        alpha = 2.0
        gamma = 2.0
        dark = (1.0 - L).pow(gamma)
        highlight = torch.sigmoid((L - 0.75) / 0.06)
        protect = 1.0 - 0.7 * highlight
        delta = (alpha * illum_map * dark * protect).clamp(0.0, 1.0)
        L_target = L + (1.0 - L) * delta

        enhanced_illum = (R * L_target).clamp(0.0, 1.0)

        # Branch 2: Color restoration
        color_correction = self.color_restore(x, f)

        # Branch 3: Final refinement with all information
        combined_features = torch.cat([f, enhanced_illum], dim=1)
        residual = self.refine(combined_features)

        # Final prediction
        pred = enhanced_illum + residual * color_correction
        pred = pred.clamp(0.0, 1.0)

        # loss = self._calculate_loss(pred, gt, illum_map) if gt is not None else None
        if gt is not None:
            loss, loss_dict = self._calculate_loss(pred, gt, illum_map)
        else:
            loss, loss_dict = None, {}

        return {
            'prediction': pred,
            'loss': loss,
            'loss_dict': loss_dict,  # 🔥 추가!
            'illum_map': illum_map,
            'color_correction': color_correction
        }

    def _calculate_loss(self, pred, gt, illum_map):
        """
        Multi-component loss for better quality
        """
        # 1. Reconstruction loss (L1 + MSE)
        l1_loss = self.l1_loss(pred, gt)
        mse_loss = self.mse_loss(pred, gt)
        recon_loss = l1_loss + 0.5 * mse_loss

        # 2. Perceptual loss (in frequency domain - efficient)
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        gt_fft = torch.fft.rfft2(gt, norm='ortho')
        freq_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(gt_fft))

        # 3. Color consistency loss
        pred_mean = pred.mean(dim=[2, 3], keepdim=True)
        gt_mean = gt.mean(dim=[2, 3], keepdim=True)
        color_loss = F.l1_loss(pred_mean, gt_mean)

        # 4. Edge preservation loss
        def get_edges(img):
            # Sobel filter
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=img.dtype, device=img.device, requires_grad=False).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-2, -1)

            edges = []
            for i in range(img.shape[1]):  # For each channel
                edge_x = F.conv2d(img[:, i:i + 1], sobel_x, padding=1)
                edge_y = F.conv2d(img[:, i:i + 1], sobel_y, padding=1)
                edges.append(torch.sqrt(edge_x ** 2 + edge_y ** 2))
            return torch.cat(edges, dim=1)

        pred_edges = get_edges(pred)
        gt_edges = get_edges(gt)
        edge_loss = F.l1_loss(pred_edges, gt_edges)

        # 5. Illumination smoothness loss (avoid over-amplification)
        illum_grad_x = illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]
        illum_grad_y = illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]
        smooth_loss = (illum_grad_x.abs().mean() + illum_grad_y.abs().mean())

        # Total loss
        total_loss = (
                1.0 * recon_loss +
              #  0.1 * freq_loss +
                0.4 * color_loss +  # 0.2 → 0.4 (2배)
                0.05 * smooth_loss
        )
        # print("Total Loss: {} | Recon Loss {} | Freq Loss {} | Color Loss {} | Smooth Loss {}".format(total_loss, recon_loss, freq_loss, color_loss, smooth_loss))

        # return total_loss

        # 개선
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'freq': freq_loss.item(),
            'color': color_loss.item(),
            'smooth': smooth_loss.item(),
        }
        return total_loss, loss_dict

# ============= Model Size Checker =============
def get_model_size(model):
    """모델 파일 크기 확인 (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 ** 2
    return size_mb


# ============= Test =============
if __name__ == "__main__":
    # 모델 생성 및 크기 확인
    model = NanoLLEv2(base_channels=24, depths=(2, 2, 3))

    # 모델 크기 체크
    size_mb = get_model_size(model)
    print(f"Model Size: {size_mb:.3f} MB")

    # Parameter 수 확인
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params:,}")

    # Forward test
    dummy_input = {
        'LQ_image': torch.randn(1, 3, 256, 256),
        'HQ_image': torch.randn(1, 3, 256, 256)
    }

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nOutput shape: {output['prediction'].shape}")
    print(f"Illumination map shape: {output['illum_map'].shape}")
    print(f"Color correction shape: {output['color_correction'].shape}")

    if output['loss'] is not None:
        print(f"Loss: {output['loss'].item():.4f}")

    # Inference speed test
    import time

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model({'LQ_image': torch.randn(1, 3, 256, 256)})

        # Measure
        start = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model({'LQ_image': torch.randn(1, 3, 256, 256)})
        end = time.time()

    avg_time = (end - start) / num_runs * 1000  # ms
    print(f"\nAverage Inference Time: {avg_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.1f}")