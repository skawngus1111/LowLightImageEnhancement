# LowLightImageEnhancement
LowLightImageEnhancement

CUDA_VISIBLE_DEVICES=3 python3 train_llie_v6.py --base_channels 20   --depths 1 1 2   --gn_groups 8   --w_ssim 0.1   --w_lpips 0.1   --w_color 1.0   --lpips_net alex   --total_iters 300000   --eval_interval 2000   --log_interval 200   --num_workers 4   --save_dir   ./runs/revised_gamma_clahe_sobel  --export_g_only --train_root dataset/EfficientLLIE/train/ --val_root dataset/EfficientLLIE/val/
