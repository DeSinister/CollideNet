import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HyCT(nn.Module):
    def __init__(self, num_frames=16, d_model=8, nhead=8, dropout=0.4, num_layers=2):
        super(HyCT, self).__init__()

        # Pre-trained ResNet50 for feature extraction
        resnet = models.resnet50(pretrained=True)
        # Removing the fully connected layer and the average pooling layer to get 3D features
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # To get (batch, 2048, 1, 1)

        # Transformer Encoder for temporal processing
        self.d_model = d_model
        self.embedding_dim = 2048  # Output dimension from ResNet50
        self.fc_resnet = nn.Linear(self.embedding_dim, d_model) 

        # Positional embeddings
        self.position_embedding = nn.Parameter(torch.zeros(num_frames, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        # Transformer encoder layer with multi-head attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        


    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        # Extract features for each frame using ResNet
        x = x.view(batch_size * num_frames, c, h, w)  # (batch * num_frames, channels, height, width)
        features = self.resnet(x)  # (batch * num_frames, 2048, height, width)
        features = self.avgpool(features).view(batch_size, num_frames, -1)
        features = self.fc_resnet(features)

        # Adding positional embeddings
        # Expand positional embeddings to (batch, num_frames, d_model)
        positional_embeddings = self.position_embedding.unsqueeze(0)  # (1, num_frames, d_model)
        features = features + positional_embeddings  # (batch, num_frames, d_model)

        # Permute for transformer (batch, num_frames, d_model) -> (num_frames, batch, d_model)
        features = features.permute(1, 0, 2)
        
        # Apply Transformer Encoder (num_frames, batch, d_model)
        transformer_out = self.transformer(features)

        # Global Max Pooling over time (batch, d_model)
        out, _ = torch.max(transformer_out, dim=0)
        
        return out


hyCT = HyCT()

if __name__ == "__main__":
    model = HyCT().cuda()
    print(model(torch.zeros(2, 16, 3, 224, 224).cuda()).shape)