import torch
import torch.nn as nn
import torch.nn.functional as F

# this works quite well and is currently #1
class UNetDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.3)  # Dropout layer after encoder1
        self.encoder2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.3)  # Dropout layer after encoder2
        self.encoder3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        #self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)

        self.decoder3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e1 = F.leaky_relu(self.encoder1(x))
        e1 = self.dropout1(e1)  # Apply dropout
        e2 = F.leaky_relu(self.encoder2(self.pool(e1)))
        e2 = self.dropout2(e2)  # Apply dropout
        e3 = F.leaky_relu(self.encoder3(self.pool(e2)))

        d3 = F.leaky_relu(self.decoder3(e3))
        d3 = F.pad(d3, (0, 1))  # Adds one unit of padding along dimension 3
        d2 = F.leaky_relu(self.decoder2(d3 + e2))  # Skip connection
        #d2 = F.pad(d2, (0, 1))  # Adds one unit of padding along dimension 3
        d1 = self.decoder1(d2 + e1)         # Skip connection
        return d1