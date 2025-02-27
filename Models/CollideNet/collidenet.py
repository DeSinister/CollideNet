import torch
import torch.nn as nn
from Models.CollideNet.hiera import hiera_base_plus_224_3_stages
from Models.CollideNet.layers.Embed import DataEmbedding_wo_temp
from Models.CollideNet.layers.SelfAttention_Family_prev import SegmentCorrelation2, SegmentCorrelation3, MultiScaleAttentionLayer
from Models.CollideNet.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        # print("PROJ")
        # print(x.shape)
        # print(stats.shape)
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs= configs
        self.spatial_model = hiera_base_plus_224_3_stages().cuda()
        self.spatial_model.head = nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth", map_location='cpu')
        self.spatial_model.load_state_dict(state_dict['model_state'], strict=False)

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiScaleAttentionLayer(
                        SegmentCorrelation2(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiScaleAttentionLayer(
                        SegmentCorrelation2(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                        configs.d_model, configs.n_heads),
                    MultiScaleAttentionLayer(
                        SegmentCorrelation3(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        p_hidden_dims=  [128, 128] # List composed of hidden dimensions for the Projector
        p_hidden_layers= 2 # Number of Layers for the Projector
        self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=self.seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=self.seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=self.seq_len)
        

    def forward(self, x, enc_self_mask = None, dec_self_mask = None, dec_enc_mask=None):
        
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x = self.spatial_model(x)
        x_enc = x.view(x_shape[0], x_shape[1], -1)
        x_dec = torch.zeros(x_shape[0], self.label_len, self.configs.enc_in)
        # decoder input
        dec_inp = torch.zeros_like(x_dec[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :configs.label_len, :], dec_inp], dim=1).float()
       


        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # print("Seasonal init", seasonal_init.shape)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # normalize
        x_raw = x_enc.clone()
        x_enc = x_enc - mean        
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean)      # B x S x E, B x 1 x E -> B x S
       

        # # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        # print("FINAL ENC: ", enc_out.shape)
        # enc_out = torch.rand(32, 30, 256)
        # dec
        # print(seasonal_init.shape)
        dec_out = self.dec_embedding(seasonal_init)
        # delta = F.pad(delta, (0, self.pred_len), mode='constant', value=0)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # print(seasonal_init.shape, dec_out.shape, enc_out.shape, delta.shape)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init, tau=tau, delta=delta)
        
        # final
        dec_out = trend_part + seasonal_part

         # De-normalization
        dec_out = dec_out * std_enc + mean

        if self.output_attention:
            return torch.squeeze(dec_out[:, -self.pred_len:, :]), attns
        else:
            return torch.squeeze(dec_out[:, -self.pred_len:, :])  # [B, L, D]

from types import SimpleNamespace
# if __name__ == "__main__":
configs = {
    'enc_in': 448,
    'dec_in': 448,
    'c_out': 448,    
    'd_model': 448, # change from 256,  
    'dropout': 0.05,
    'e_layers': 2,
    'factor': 1,
    'label_len': 0,
    'pred_len': 1,
    'seq_len': 16,
    'moving_avg': 7,
    'n_heads': 8,
    'num_workers': 1,
    'output_attention': False,
    'activation': 'gelu',  
    'd_layers': 1,     
    'd_ff': 2048  
    }
print(configs.values())
configs = SimpleNamespace(**configs)
collidenet = Model(configs=configs).cuda()
bs=2
print("param", sum(p.numel() for p in collidenet.parameters() if p.requires_grad))
print("param", sum(p.numel() for p in collidenet.spatial_model.parameters() if p.requires_grad))