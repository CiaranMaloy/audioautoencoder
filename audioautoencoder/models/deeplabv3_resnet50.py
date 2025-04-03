import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter

# Step 1: Load the original ResNet50 backbone with pretrained weights
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Step 2: Modify the first convolution (conv1) to accept 4 channels
old_conv = resnet.conv1
new_conv = nn.Conv2d(
    9,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)
with torch.no_grad():
    # Copy the weights for the first 3 channels
    new_conv.weight[:, :3, :, :] = old_conv.weight
    # For the 4th channel, use the mean of the original weights
    new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
resnet.conv1 = new_conv

# Step 3: Create a new backbone using IntermediateLayerGetter
# We extract the layers needed by DeepLabV3 (by default, "layer1" and "layer4")
return_layers = {'layer1': 'low_level_features', 'layer4': 'out'}
backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

# Step 4: Build the DeepLabV3 model using the custom backbone
# Here we use a custom classifier head; adjust num_classes if needed
num_classes = 21  # initial number for COCO/VOC; you'll change this below
model = torch.hub.load("pytorch/vision:v0.13.1", "deeplabv3_resnet50", pretrained=False)
model.backbone = backbone

# Optionally, if you want to modify the classifier output to 4 channels (for a 4-channel mask):
model.classifier = DeepLabHead(2048, 256)  # Create a new classifier head
# Replace the last convolution to output 4 channels instead of the original number
model.classifier[4] = nn.Conv2d(256, 4, kernel_size=1)

# Example custom wrapper (e.g., for spectrogram masking)
class SpectrogramMaskingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            mask = torch.sigmoid(output['out'])
        else:
            mask = torch.sigmoid(output)
        denoised = x[:, :4] * mask[:, :4]  # Assuming you want elementwise multiplication over channels
        return denoised #, mask

# Instantiate your custom model
spectrogram_model = SpectrogramMaskingModel(model)

# Example loss function combining reconstruction loss and sparsity loss
def loss_fn(denoised, target, mask=None, alpha=0.8, beta=0.2):
    l1_loss = F.l1_loss(denoised, target)
    if mask is not None:
        sparsity_loss = torch.mean(torch.abs(mask))
        return alpha * l1_loss + beta * sparsity_loss
    return l1_loss

# Stage 1: Freeze all layers except the modified conv1 and the new classifier layer
for param in model.parameters():
    param.requires_grad = True
for param in model.backbone.conv1.parameters():
    param.requires_grad = True
for param in model.classifier[4].parameters():
    param.requires_grad = True

# Now, you can train your model as needed.
