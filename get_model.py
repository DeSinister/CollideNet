import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    
def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:]
			state_dict[new_key] = state_dict.pop(old_key)
		else:
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)
     

def init_from_pretrain_(module, pretrained, init_module):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrained)
    else:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    if init_module == 'transformer':
        replace_state_dict(state_dict)
    elif init_module == 'cls_head':
        replace_state_dict(state_dict)
    else:
        raise TypeError(f'pretrained weights do not include the {init_module} module')
    msg = module.load_state_dict(state_dict, strict=False)
    return msg

# Create a class for the Model with Downstream task.
class VideoEncoder(nn.Module):
    def __init__(self, model_type='VideoSwin', clamp_limit=6.0):
        super().__init__() 
        self.clamp_limit = clamp_limit
        embed_dims = None
        head_layer_embed = 64
        # Load the appropriate video transformer model and set the embedding dimension
        if model_type == 'CNN-RNN':  
            from Models.CNNRNN.cnn_rnn import cnn_rnn
            self.vid_trans = cnn_rnn  # Load CNN-RNN model
            embed_dims = 8  # Set embedding dimension
            head_layer_embed = 8
        elif model_type == 'C3D':  
            from Models.C3D.c3d import c3d
            self.vid_trans = c3d  # Load C3D model
            embed_dims = 487  # Set embedding dimension
        elif model_type == 'ResNet3D':  
            self.vid_trans = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True) # Load ResNet 3D model
            self.vid_trans.blocks[-1].proj = nn.Identity()  
            embed_dims = 2048  # Set embedding dimension
        elif model_type == 'VGG16':  
            from Models.VGG16.vgg_6f import vgg_6f
            self.vid_trans = vgg_6f  # Load VGG-16 model
            embed_dims = 4704  # Set embedding dimension
        elif model_type == 'X3D':  
            self.vid_trans = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True) # Load X3D 3D model
            self.vid_trans.blocks[-1].proj = nn.Identity()  
            embed_dims = 2048  # Set embedding dimension
        elif model_type == 'ViViT':  
            from Models.ViViT.video_transformer import ViViT
            self.vid_trans = ViViT(num_frames=16,
                  img_size=224,
                  patch_size=16,
                  embed_dims=768,
                  in_channels=3,
                  attention_type='fact_encoder',
                  return_cls_token=True)    # Load ViViT model
            init_from_pretrain_(self.vid_trans, "vivit_pytorch\\vivit_model.pth", init_module='transformer')
            embed_dims = 768  # Set embedding dimension
        if model_type == 'TimeSformer':  
            from Models.timesformer.models.vit import TimeSformer
            self.vid_trans = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='timesformer\\TimeSformer_divST_8x32_224_K400.pyth') # Load TimeSformer model
            self.vid_trans.model.head = nn.Identity()
            embed_dims = 768  # Set embedding dimension
        elif model_type == 'VideoSwin':
            from Models.Video_SwinTransformer.swin import swin_base
            self.vid_trans = swin_base  # Load Video Swin Transformer model
            embed_dims = 1024  # Set embedding dimension
        elif model_type == 'Li3D':  
            from Models.Li3D.li3d import li3d
            self.vid_trans = li3d  # Load Li3D model
            embed_dims = 1920  # Set embedding dimension
        elif model_type == 'HyCT':  
            from Models.HyCT.hyCT import hyCT
            self.vid_trans = hyCT  # Load CTH model
            embed_dims = 8  # Set embedding dimension
            head_layer_embed = 8
        elif model_type == 'VideoFocalNets':  
            from Models.Video_FocalNets.video_focalnet import videofocalnet_b
            self.vid_trans = videofocalnet_b  # Load Video FocalNet model
            embed_dims = 1024  # Set embedding dimension
        elif model_type == 'VidNeXt':  
            from Models.vidnext.vidnext import vidnext
            self.vid_trans = vidnext  # Load VidNeXt model
            embed_dims = 1024  # Set embedding dimension
        elif model_type == 'CollideNet': 
            from Models.CollideNet.collidenet import collidenet
            self.vid_trans = collidenet  # Load CollideNet model
            embed_dims = 448  # Set embedding dimension
        

        # Define the final regression head for predicting Time-to-Collision (TTC)
        self.final_layer = nn.Sequential(
            nn.ReLU(),  # Apply ReLU activation
            nn.Linear(embed_dims, head_layer_embed),  # Linear layer to reduce embedding size
            nn.ReLU(),  # Apply ReLU activation
            nn.Dropout(),  # Apply dropout for regularization
            nn.Linear(head_layer_embed, 1)  # Final linear layer to output a single scalar value
        )
    
    
    # Forward Method
    def forward(self, x):
        # Extract video features using the selected transformer model
        x = self.vid_trans(x)       
        # Pass features through the regression head to predict the TTC score
        x = self.final_layer(x)
        x = torch.clamp(x, min=0.0, max=self.clamp_limit)
        return x