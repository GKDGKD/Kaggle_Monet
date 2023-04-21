import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class downsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, 
                 stride=2, padding=1, apply_instancenorm=True):
        super().__init__()
        self.net = nn.ModuleList([nn.Conv2d(in_channel, out_channel, kernel_size, stride, 
                                            padding, bias=False)])
        
        if apply_instancenorm:
            self.net.append(nn.InstanceNorm2d(out_channel))

        self.net.append(nn.LeakyReLU())

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class upsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU() # inplace会覆盖原内存
        )

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        # Input: [bs, 3， 256, 256]
        self.down_stack = nn.ModuleList([
            downsample(in_channels, 64, apply_instancenorm=False), # [bs, 64, 128, 128]  
            downsample(64,  128),       # [bs, 128, 64, 64]
            downsample(128, 256),       # [bs, 256, 32, 32]
            downsample(256, 512),       # [bs, 512, 16, 16]
            downsample(512, 512),       # [bs, 512, 8, 8]
            downsample(512, 512),       # [bs, 512, 4, 4]
            downsample(512, 512),       # [bs, 512, 2, 2]
            downsample(512, 512, apply_instancenorm=False),       # [bs, 512, 1, 1]
        ])

        self.up_stack = nn.ModuleList([
            upsample(512, 512),  # (bs, 1024, 2, 2)  拼接后的shape
            upsample(1024, 512), # (bs, 1024, 4, 4)  拼接后 in_channels = 上一层的 out_channels * 2
            upsample(1024, 512), # (bs, 1024, 8, 8)
            upsample(1024, 512), # (bs, 1024, 16, 16)
            upsample(1024, 256), # (bs, 512, 32, 32)
            upsample(512, 128),  # (bs, 256, 64, 64)
            upsample(256, 64),   # (bs, 128, 128, 128)
        ])

        self.last = upsample(128, 3) 

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        out = self.last(x)

        return out 
    
class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(Discriminator, self).__init__()  
        
        in_channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.model = nn.ModuleList(
            [   # input: [bs, 3, 256, 256]
                downsample(in_channels, 64, apply_instancenorm=False), # [bs, 64, 128, 128] , apply_instancenorm=False
                downsample(64, 128),        # [bs, 128, 64, 64]
                downsample(128, 256),       # [bs, 256, 32, 32]
                downsample(256, 512),       # [bs, 512, 16, 16]
                nn.ZeroPad2d((1, 0, 1, 0)), # [bs, 512, 17, 17]
                nn.Conv2d(512, 1, 4, padding=1) # [bs, 1, 16, 16]
            ]
    )
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            
        return x