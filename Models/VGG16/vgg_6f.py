
import torch.nn as nn 
import torch
import torchvision.models as models


class VGGNet(nn.Module):
    
    def __init__(self):
        super(VGGNet, self).__init__()
        self.rgb_net = self.get_vgg_features()
        
        kernel_size = 3 
        padding = int((kernel_size - 1)/2)
        self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
        #self.conv_bn = nn.BatchNorm2d(16)
        ## input_channels, output_channels, kernel_size, stride, padding, bias
        self.feature_size = 16*7*7*6
        #self.feature_size = 16*7*7*3 
        self.final_layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.feature_size, 2048),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2048, 1)
        #nn.Sigmoid()
        #nn.Softmax()  ## If loss function uses Softmax  
        )
        
    def forward(self, rgb): ## sequence of four images - last index is latest 
        four_imgs = []
        for i in range(rgb.shape[1]):
            img_features = self.rgb_net(rgb[:,i,:,:,:])
            channels_reduced = self.conv_layer(img_features)
            img_features = channels_reduced.view((-1, 16*7*7))
            four_imgs.append(img_features)
        concat_output = torch.cat(four_imgs, dim = 1)
        out = self.final_layer(concat_output)
        return out
#         return concat_output

        
    def get_vgg_features(self):

        modules = list(vgg_model.children())[:-1]
        vgg16 = nn.Sequential(*modules)
        
        ## Uncommented this to let it fine-tune on my model 
        # for p in vgg16.parameters():
        #     p.requires_grad = False 
        
        return vgg16.type(torch.Tensor)


def load_vgg_voc_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    vgg_model.load_state_dict(checkpoint_dict)

vgg_model = models.vgg16(pretrained=True)
num_final_in = vgg_model.classifier[-1].in_features
vgg_model.classifier[-1] = nn.Linear(num_final_in, 20)
# model_path = '<Path to vgg_on_voc800>'
# load_vgg_voc_weights(model_path)



class VGGNet(nn.Module):
    
    def __init__(self):
        super(VGGNet, self).__init__()
        self.rgb_net = self.get_vgg_features()
        
        kernel_size = 3 
        padding = int((kernel_size - 1)/2)
        self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
        #self.conv_bn = nn.BatchNorm2d(16)
        ## input_channels, output_channels, kernel_size, stride, padding, bias
        self.feature_size = 16*7*7*6
        #self.feature_size = 16*7*7*3 
    def forward(self, rgb): ## sequence of four images - last index is latest 
        four_imgs = []
        for i in range(rgb.shape[1]):
            img_features = self.rgb_net(rgb[:,i,:,:,:])
            channels_reduced = self.conv_layer(img_features)
            img_features = channels_reduced.view((-1, 16*7*7))
            four_imgs.append(img_features)
        out = torch.cat(four_imgs, dim = 1)
        return out
        # return concat_output
        
    def get_vgg_features(self):

        modules = list(vgg_model.children())[:-1]
        vgg16 = nn.Sequential(*modules)
        
        return vgg16.type(torch.Tensor)



vgg_6f= VGGNet()
if __name__ == "__main__":
    model = vgg_6f.cuda()
    print(model(torch.zeros(4, 6, 3, 224, 224).cuda()).shape)