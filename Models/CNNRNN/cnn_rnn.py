import torch
import torch.nn as nn
import torchvision.models as models

class InceptionGRUModel(nn.Module):
    def __init__(self):
        super(InceptionGRUModel, self).__init__()
        
        # Load pre-trained Inception v3 and remove the fully connected layers
        self.inception_v3 = models.inception_v3(pretrained=True)
        self.inception_v3.fc = nn.Identity()
        
        # Define Global Average Pooling (to reduce spatial dimensions)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (batch_size, 2048, 1, 1)
        
        # First GRU layer with 16 neurons, followed by a second GRU with 8 neurons
        self.gru = nn.GRU(
            input_size=2048,  # Inception v3 feature size after pooling
            hidden_size=16,   # First GRU layer with 16 neurons
            num_layers=1,     # One GRU layer at a time
            batch_first=True
        )
        
        self.gru2 = nn.GRU(
            input_size=16,    # Output from the first GRU
            hidden_size=8,    # Second GRU layer with 8 neurons
            num_layers=1,
            batch_first=True
        )
        
        

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape (batch_size, seq_len, channels, height, width) -> (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Pass the entire batch of frames through Inception v3 feature extractor
        with torch.no_grad():  # Do not train Inception v3
            # print(x.shape)
            if self.training:
                features = self.inception_v3(x).logits  # (batch_size * seq_len, 2048, h, w)
            else:
                features = self.inception_v3(x)
            # print(features.shape)
            # features = self.global_avg_pool(features)  # Apply Global Average Pooling: (batch_size * seq_len, 2048, 1, 1)
            features = features.view(batch_size * seq_len, -1)  # Flatten: (batch_size * seq_len, 2048)
        
        # Reshape back to (batch_size, seq_len, 2048) for the GRU
        features = features.view(batch_size, seq_len, -1)
        
        # First GRU layer
        gru_out, _ = self.gru(features)  # (batch_size, seq_len, 16)
        
        # Second GRU layer
        gru_out2, _ = self.gru2(gru_out)  # (batch_size, seq_len, 8)
        
        # Take the output from the last time step
        out = gru_out2[:, -1, :]  # (batch_size, 8)
        
        return out



cnn_rnn = InceptionGRUModel()
if __name__ == "__main__":
    cnn_rnn = InceptionGRUModel()
    cnn_rnn = cnn_rnn.cuda()
    print(cnn_rnn(torch.zeros(2, 16, 3, 299, 299).cpu()).shape)

