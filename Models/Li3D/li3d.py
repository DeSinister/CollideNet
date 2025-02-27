import torch
import torch.nn as nn
import torch.nn.functional as F

class Li3D(nn.Module):
    def __init__(self):
        super(Li3D, self).__init__()

        # First block: 32 filters, kernel size (3,3,3), stride 1, ReLU
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv1b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(3,3,3), padding=1)
        self.dropout1 = nn.Dropout3d(p=0.25)
        self.norm1 = nn.BatchNorm3d(32)

        # Second block: 64 filters, kernel size (3,3,3), stride 1, ReLU
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv2b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), padding=1)
        self.dropout2 = nn.Dropout3d(p=0.25)
        self.norm2 = nn.BatchNorm3d(64)

        # Third block: 128 filters, kernel size (3,3,3), stride 1, ReLU
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(3,3,3), padding=1)
        self.dropout3 = nn.Dropout3d(p=0.25)
        self.norm3 = nn.BatchNorm3d(128)


    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        
        # Block 1
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        x = self.dropout1(x)
        x = self.norm1(x)

        # Block 2
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        x = self.dropout2(x)
        x = self.norm2(x)

        # Block 3
        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        # print(f"After pool3: {x.shape}")
        x = self.dropout3(x)
        x = self.norm3(x)

        # Flatten the 3D output to 1D
        x = x.view(x.size(0), -1)
        # print(f"After flattening: {x.shape}")

        return x


li3d  = Li3D()

if __name__ == "__main__":
    model = Li3D()
    print(model(torch.zeros(2, 3, 16, 128, 72).cpu()).shape)
