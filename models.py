
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
import os
from torch.utils.data import random_split
from PIL import Image
import time
import torchvision.transforms as transforms
from pytorch_msssim import ssim
from matplotlib import pyplot as plt
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter




'''
The code for AdaIN was borrowed from this individual, who re-wrote the original code in python (was Lua)
https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py

Original: https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/AdaptiveInstanceNormalization.lua

When I say "Original," I'm reffering to the code that the authors of the paper that this is based of wrote for their paper
'''



# Calculates the Gradient magnitude of the passed image and returns it, does so by performing a directional sobel convolution in the x and y direction and then using those as inputs for the gradient magnitude function
def sobel_filter(image):

    # setup the kernals for direction sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # resize to cover all channels of the passed image
    channels = image.size(1)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1).to(image.device)

    # apply sobel in x and y direction
    edges_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    edges_y = F.conv2d(image, sobel_y, padding=1, groups=channels)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)

    edges = torch.clamp(edges, 0.0, 1.0)

    
    # normalize the gradient magnitude and return
    #edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6) 
    return edges


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat



def AdaIn(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


'''
    This code for CBAM was borrowed from this github:https://github.com/Jongchan/attention-module/blob/c06383c514ab0032d044cc6fcd8c8207ea222ea7/MODELS/cbam.py#L84

    It's the official implementation from the researchers who first proposed it
'''


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Get the pretrained vgg19 model
        vgg = models.vgg19(pretrained=True).features

        # Define layers to extract features at specific layers
        self.model = nn.Sequential(*[vgg[i] for i in range(len(vgg))])  
        
        # Freeze weights so we don't change anything we didn't mean to 
        for param in self.model.parameters():
            param.requires_grad = False

        # These need to be calculated seperatly, as for some loss calculation tasks we care about the higher level feature maps
        self.relu1_1 = self.model[:2]
        self.relu2_1 = self.model[2:7]
        self.relu3_1 = self.model[7:12]
        self.relu4_1 = self.model[12:21]
        self.relu5_1 = self.model[21:28]

    # Generate feature maps, X is the input image, returns relu1_1 - relu4_1 feature maps
    def forward(self, x):

        feat1_1 = self.relu1_1(x)
        feat2_1 = self.relu2_1(feat1_1)
        feat3_1 = self.relu3_1(feat2_1)
        feat4_1 = self.relu4_1(feat3_1)

        return feat1_1,feat2_1,feat3_1,feat4_1

# Custom decoder model, takes the processed feature map and reconstructs it back to the original image space as it goes along
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Helper to create a block: Reflection Pad -> Conv -> ReLU
        # ReflectionPad2d(1) mirrors the edge pixels instead of adding black (0) borders
        def dec_block(in_c, out_c):
            return nn.Sequential(
                nn.ReflectionPad2d(1), 
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )

        # 4_1 -> 3_1
        self.layer4 = dec_block(512, 256)
         
        # 3_1 -> 2_1
        self.layer3 = dec_block(256, 128)
        
        # 2_1 -> 1_1
        self.layer2 = dec_block(128, 64)
        
        # 1_1 -> Image
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() 
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Input: 512 channels (from Encoder 4_1)
        
        # Upsample -> Conv Block
        x = self.layer4(x)      # 512 -> 256
        x = self.upsample(x)
        
        x = self.layer3(x)      # 256 -> 128
        x = self.upsample(x)
        
        x = self.layer2(x)      # 128 -> 64
        x = self.upsample(x)
        
        x = self.layer1(x)      # 64 -> 3 (RGB)
        
        return x
    


class DeepDecoder(nn.Module):
    def __init__(self):
        super(DeepDecoder, self).__init__()

        # Helper to create a block: Reflection Pad -> Conv -> ReLU
        # ReflectionPad2d(1) mirrors the edge pixels instead of adding black (0) borders
        def dec_block(in_c, out_c):
            return nn.Sequential(
                nn.ReflectionPad2d(1), 
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )

        # 4_1 -> 3_1 (1 conv: 512 -> 256)
        self.layer4 = dec_block(512, 256)
         
        # 3_1 -> 2_1 (4 convs: 256 -> 256 -> 256 -> 256 -> 128)
        self.layer3 = nn.Sequential(
            dec_block(256, 256),
            dec_block(256, 256),
            dec_block(256, 256),
            dec_block(256, 128),
        )
        
        # 2_1 -> 1_1 (2 convs: 128 -> 128 -> 64)
        self.layer2 = nn.Sequential(
            dec_block(128, 128),
            dec_block(128, 64),
        )
        
        # 1_1 -> Image (2 convs: 64 -> 64 -> 3)
        self.layer1 = nn.Sequential(
            dec_block(64, 64),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid() 
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Input: 512 channels (from Encoder 4_1)
        
        x = self.layer4(x)      # 512 -> 256
        x = self.upsample(x)
        
        x = self.layer3(x)      # 256 -> 128
        x = self.upsample(x)
        
        x = self.layer2(x)      # 128 -> 64
        x = self.upsample(x)
        
        x = self.layer1(x)      # 64 -> 3 (RGB)
        
        return x



    
# Put the models together into a single model
class StyleTransferModel(nn.Module):
    def __init__(self, alpha=1.0, standardize_encoder_inputs=True, DeepDecoderTrue=False):
        super(StyleTransferModel, self).__init__()

        self.encoder = Encoder().eval()
        self.cbam = CBAM(512)
        self.decoder = DeepDecoder() if DeepDecoderTrue else Decoder()

        #alpha can be altered to adjust style application strength post training (0-1)
        self.alpha = alpha
        self.standardize_encoder_inputs = standardize_encoder_inputs

        if standardize_encoder_inputs:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    # Process the content and style images
    def forward(self, content, style):

        if self.standardize_encoder_inputs:
            content = (content - self.mean) / self.std
            style = (style - self.mean) / self.std
        
        # Pass content and style through encoder to get  feature maps
        content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1 = self.encoder(content)
        style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1 = self.encoder(style)

        # Place these in lists so we can access them easily later
        content_feats = [content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1]
        style_feats = [style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1]

        # Apply CBAM, add the out put back into the original for the skip connection
        attention4_1 = self.cbam(content_feat4_1)

        # Skip connection
        attention_boosted_4_1 = content_feat4_1+attention4_1

        # Perform adaptive instance normalization with to fuse style into content
        fused_feat4_1 = AdaIn(attention_boosted_4_1,style_feat4_1)

        # Just scales the degree of which the style is applied, if one it remains the same
        fused_feat4_1 = self.alpha * fused_feat4_1 + (1 - self.alpha) * content_feat4_1
    

        # decode the image
        generated_image = self.decoder(fused_feat4_1)
        
        return generated_image,style_feats,content_feats

    
# Custom Loss Class, will be used for both model cases, but the experiment with no ssim will set ssim lambda to 0

'''
The code for the style and content loss was borrowed from this individual, who re-wrote the original code in python (was Lua)
https://github.com/naoto0804/pytorch-AdaIN/blob/master/net.py

Original: https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/ContentLossModule.lua

Specifically, the code for calc_content_loss and calc_style_loss were used
'''



class StyleTransferLoss(nn.Module):
    def __init__(self, vgg_encoder, lambda_c=1, lambda_s=10, lambda_ssim=1):

        super(StyleTransferLoss, self).__init__()
        self.vgg_encoder = vgg_encoder 
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_ssim = lambda_ssim
        self.mse_loss = nn.MSELoss()


        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
    

        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # Style loss based on mean and variance
        assert input.size() == target.size(), f"Style size mismatch! {input.size()} vs {target.size()}"
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    # Actual loss calcualtion, takes the generated image, as well as the content image, and the feature maps for content and style
    def forward(self, generated_image, content,content_feats, style_feats):

        # encode the generated image for loss calculation
        gen_norm = (generated_image - self.mean) / self.std
        gen_feats = self.vgg_encoder(gen_norm)
        #gen_feats = self.vgg_encoder(generated_image)

        
        # calculate content loss (last feature map for structure preservation)
        content_loss = self.calc_content_loss(gen_feats[-1], content_feats[-1])
        
        # calculate style loss (mean and variance alignment for each layer)
        style_loss = 0
        for gen_feat, style_feat in zip(gen_feats, style_feats):
            style_loss += self.calc_style_loss(gen_feat, style_feat)

        # Sobel-SSIM edge loss
        sobel_gen = sobel_filter(generated_image)
        sobel_content = sobel_filter(content)

        edge_loss = 1 - ssim(sobel_gen, sobel_content, data_range=1.0, size_average=True)

        # Combine the losses with weights
        total_loss = (self.lambda_c * content_loss +
                      self.lambda_s * style_loss +
                      self.lambda_ssim * edge_loss)

        return total_loss, self.lambda_c * content_loss, self.lambda_ssim*edge_loss,self.lambda_s*style_loss


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Permute from (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)


class PatchGAN(nn.Module):
    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        super(PatchGAN, self).__init__()
        
        # Conv ->  LeakyReLU
        sequence = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Gradually increase filter size
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                LayerNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        # Stride 1 to retain spatial resolution for the final patch map
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            LayerNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Final output layer (1 channel prediction map)
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
    

def gradient_penalty(discriminator, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    pred = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty