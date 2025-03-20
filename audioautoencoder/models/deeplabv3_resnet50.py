from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch

# Load pre-trained model
model = deeplabv3_resnet50(pretrained=True)

# Modify the input layer to accept 4 channels instead of 3
original_conv = model.backbone.layer0.conv1
new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Initialize the new conv layer with weights from the pre-trained model
# For the first 3 channels, copy weights from the original layer
# For the 4th channel, initialize with mean of the original weights
with torch.no_grad():
    new_conv.weight[:, :3, :, :] = original_conv.weight
    new_conv.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

# Replace the original conv layer with the new one
model.backbone.layer0.conv1 = new_conv

# Modify the output layer to produce a 4-channel mask
model.classifier[4] = nn.Conv2d(256, 4, kernel_size=1)

class SpectrogramMaskingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model  # The modified DeepLabV3 model
        
    def forward(self, x):
        # x is your 4-channel spectrogram
        features = self.model(x)
        mask = torch.sigmoid(features['out'])  # Convert to probability values between 0-1
        
        # Apply the mask to your input
        denoised = x * mask
        
        return denoised, mask  # Return both denoised spectrogram and mask
    
# Loss function: combination of L1 loss and possibly spectral loss
def loss_fn(denoised, target, mask=None, alpha=0.8, beta=0.2):
    # Direct reconstruction loss
    l1_loss = F.l1_loss(denoised, target)
    
    # Optional: add mask sparsity loss to encourage cleaner masks
    if mask is not None:
        sparsity_loss = torch.mean(torch.abs(mask))
        return alpha * l1_loss + beta * sparsity_loss
    
    return l1_loss

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        
        optimizer.zero_grad()
        denoised, mask = model(inputs)
        
        loss = loss_fn(denoised, targets, mask)
        loss.backward()
        optimizer.step()

# should also train different layers differently
# Stage 1: Train only the modified layers
for param in model.parameters():
    param.requires_grad = False
    
model.backbone.layer0.conv1.requires_grad = True
model.classifier[4].requires_grad = True

# Train for a few epochs

# Stage 2: Fine-tune the entire model
for param in model.parameters():
    param.requires_grad = True

# Train for additional epochs