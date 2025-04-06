import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="resnet18",      # Lightweight backbone
    encoder_weights=None,         # No pretrained weights since we're not using 3 RGB channels
    in_channels=4,                # 👈 Accept 4-channel input (e.g., stacked spectrograms)
    classes=4                     # 👈 Output 4-channel denoised output
)