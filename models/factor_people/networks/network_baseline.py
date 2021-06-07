import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import math

class RelightNet(nn.Module):
    
    def __init__(self, num_group=16):
        super(RelightNet,self).__init__()

        self.conv00 = nn.Sequential(
                        nn.Conv2d(3, 29, 7, stride=1, padding=3),
                        nn.PReLU()) # 512 -> 512
        self.conv01 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 512 -> 256

        self.conv10 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.conv11 = nn.Sequential( 
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  # 256 -> 128

        self.conv20 = nn.Sequential( 
                        nn.Conv2d(128, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  
        self.conv21 = nn.Sequential( 
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  # 128 -> 64

        self.conv30 = nn.Sequential(
                        nn.Conv2d(256, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  
        self.conv31 = nn.Sequential(
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  # 64 -> 32

        self.conv40 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) 
        self.conv41 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  
        self.conv42 = nn.Sequential(
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())

        self.conv43 = nn.Sequential(
                        nn.Conv2d(512, 2048, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=2048),
                        nn.Softplus())
        

        self.unconv44 = nn.Sequential(
                        nn.Conv2d(512*3, 512, 1, stride=1, padding=0),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) # 32

        self.unconv43 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv42 = nn.Sequential(
                        nn.ConvTranspose2d(768, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv41 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv40 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU()) # 32 -> 64

        self.unconv31 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv30 = nn.Sequential(
                        nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU()) # 64 -> 128

        self.unconv21 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())
        self.unconv20 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 128 -> 256

        self.unconv11 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.unconv10 = nn.Sequential(
                        nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=32),
                        nn.PReLU()) # 256 -> 512

        self.unconv00 = nn.Sequential(
                        nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
                        nn.Sigmoid()) # 256 -> 512
        
    def forward(self, input_human_scene, target_light):

        bz = input_human_scene.shape[0]

        x00 = self.conv00(input_human_scene)
        x00 = torch.cat([x00, input_human_scene], dim=1)
        x10 = self.conv01(x00)
        x11 = self.conv10(x10)
        
        x20 = self.conv11(x11)
        x21 = self.conv20(x20)

        x30 = self.conv21(x21)
        x31 = self.conv30(x30)

        x40 = self.conv31(x31)
        x41 = self.conv40(x40)
        x42 = self.conv41(x41)
        x43 = self.conv42(x42)
        x50 = self.conv43(x43)
        light_map = x50[:,:512*3,:,:].view(x50.shape[0], 3, 512, x50.shape[2], x50.shape[3])
        confidence_map = x50[:,512*3:,:,:].view(x50.shape[0], 1, 512, x50.shape[2], x50.shape[3])

        light = light_map * confidence_map
        light = torch.mean(light.view(light.shape[0], light.shape[1], light.shape[2], -1), dim=3)

        est_source_light = light.view(light.shape[0], light.shape[1], 16, 32)

        target_light_view = target_light.contiguous().view(bz, -1, 1, 1)
        y43 = self.unconv44(target_light_view).repeat(1, 1, 48, 32)
        # print(y43.shape)
        y42 = self.unconv43(y43)
        # print(y42.shape)
        y41 = self.unconv42(torch.cat([y42, x42], dim=1))
        # print(y41.shape)
        y40 = self.unconv41(torch.cat([y41, x41], dim=1))
        # print(y40.shape)

        y31 = self.unconv40(torch.cat([y40, x40], dim=1))
        # print(y31.shape)
        y30 = self.unconv31(torch.cat([y31, x31], dim=1))
        # print(y30.shape)

        y21 = self.unconv30(torch.cat([y30, x30], dim=1))
        # print(y21.shape)
        y20 = self.unconv21(torch.cat([y21, x21], dim=1))
        # print(y20.shape)

        y11 = self.unconv20(torch.cat([y20, x20], dim=1))
        # print(y11.shape)
        y10 = self.unconv11(torch.cat([y11, x11], dim=1))
        # print(y10.shape)

        y00 = self.unconv10(torch.cat([y10, x10], dim=1))

        output = self.unconv00(torch.cat([y00, x00], dim=1))

        return output, est_source_light


class ShadowUnet(nn.Module):
    
    def __init__(self, num_group=1):
        super(
            ShadowUnet,self).__init__()

        self.conv00 = nn.Sequential(
                        nn.Conv2d(1, 31, 7, stride=1, padding=3),
                        nn.PReLU()) # 512 -> 512
        self.conv01 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 512 -> 256

        self.conv10 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.conv11 = nn.Sequential( 
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  # 256 -> 128

        self.conv20 = nn.Sequential( 
                        nn.Conv2d(128, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  
        self.conv21 = nn.Sequential( 
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  # 128 -> 64

        self.conv30 = nn.Sequential(
                        nn.Conv2d(256, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  
        self.conv31 = nn.Sequential(
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  # 64 -> 32

        self.conv40 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) 
        self.conv41 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  
        self.conv42 = nn.Sequential(
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())

        self.conv43 = nn.Sequential(
                        nn.Conv2d(512, 2048, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=2048),
                        nn.Softplus())
        

        self.unconv44 = nn.Sequential(
                        nn.Conv2d(512*3, 512, 1, stride=1, padding=0),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) # 32

        self.unconv43 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv42 = nn.Sequential(
                        nn.ConvTranspose2d(768, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv41 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv40 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU()) # 32 -> 64

        self.unconv31 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv30 = nn.Sequential(
                        nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU()) # 64 -> 128

        self.unconv21 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())
        self.unconv20 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 128 -> 256

        self.unconv11 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.unconv10 = nn.Sequential(
                        nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=32),
                        nn.PReLU()) # 256 -> 512

        self.unconv00 = nn.Sequential(
                        nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=1),
                        nn.Sigmoid()) # 256 -> 512
        
    def forward(self, input_human_mask, target_light):

        bz = input_human_mask.shape[0]

        x00 = self.conv00(input_human_mask)
        x00 = torch.cat([x00, input_human_mask], dim=1)
        x10 = self.conv01(x00)
        x11 = self.conv10(x10)
        
        x20 = self.conv11(x11)
        x21 = self.conv20(x20)

        x30 = self.conv21(x21)
        x31 = self.conv30(x30)

        x40 = self.conv31(x31)
        x41 = self.conv40(x40)
        x42 = self.conv41(x41)

        target_light_view = target_light.view(bz, -1, 1, 1)
        y43 = self.unconv44(target_light_view).repeat(1, 1, 48, 32)
        # print(y43.shape)
        y42 = self.unconv43(y43)
        # print(y42.shape)
        y41 = self.unconv42(torch.cat([y42, x42], dim=1))
        # print(y41.shape)
        y40 = self.unconv41(torch.cat([y41, x41], dim=1))
        # print(y40.shape)

        y31 = self.unconv40(torch.cat([y40, x40], dim=1))
        # print(y31.shape)
        y30 = self.unconv31(torch.cat([y31, x31], dim=1))
        # print(y30.shape)

        y21 = self.unconv30(torch.cat([y30, x30], dim=1))
        # print(y21.shape)
        y20 = self.unconv21(torch.cat([y21, x21], dim=1))
        # print(y20.shape)

        y11 = self.unconv20(torch.cat([y20, x20], dim=1))
        # print(y11.shape)
        y10 = self.unconv11(torch.cat([y11, x11], dim=1))
        # print(y10.shape)

        y00 = self.unconv10(torch.cat([y10, x10], dim=1))

        output = self.unconv00(torch.cat([y00, x00], dim=1))

        return output

class RelightNet_v2(nn.Module):
    
    def __init__(self, num_group=1):
        super(RelightNet_v2,self).__init__()

        self.conv00 = nn.Sequential(
                        nn.Conv2d(3, 29, 7, stride=1, padding=3),
                        nn.PReLU()) # 512 -> 512
        self.conv01 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 512 -> 256

        self.conv10 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.conv11 = nn.Sequential( 
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  # 256 -> 128

        self.conv20 = nn.Sequential( 
                        nn.Conv2d(128, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  
        self.conv21 = nn.Sequential( 
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  # 128 -> 64

        self.conv30 = nn.Sequential(
                        nn.Conv2d(256, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  
        self.conv31 = nn.Sequential(
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  # 64 -> 32

        self.conv40 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) 
        self.conv41 = nn.Sequential( 
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  
        self.conv42 = nn.Sequential(
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())

        self.conv43 = nn.Sequential(
                        nn.Conv2d(512, 2048, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=2048),
                        nn.Softplus())
        

        self.unconv44 = nn.Sequential(
                        nn.Conv2d(512*3, 512, 1, stride=1, padding=0),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) # 32

        self.unconv43 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv42 = nn.Sequential(
                        nn.ConvTranspose2d(768, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv41 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv40 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU()) # 32 -> 64

        self.unconv31 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.unconv30 = nn.Sequential(
                        nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU()) # 64 -> 128

        self.unconv21 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())
        self.unconv20 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 128 -> 256

        self.unconv11 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.unconv10 = nn.Sequential(
                        nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=32),
                        nn.PReLU()) # 256 -> 512

        self.unconv00 = nn.Sequential(
                        nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=3),
                        nn.Sigmoid()) # 256 -> 512
        
    def forward(self, input_human_scene, target_light):

        bz = input_human_scene.shape[0]

        x00 = self.conv00(input_human_scene)
        x00 = torch.cat([x00, input_human_scene], dim=1)
        x10 = self.conv01(x00)
        x11 = self.conv10(x10)
        
        x20 = self.conv11(x11)
        x21 = self.conv20(x20)

        x30 = self.conv21(x21)
        x31 = self.conv30(x30)

        x40 = self.conv31(x31)
        x41 = self.conv40(x40)
        x42 = self.conv41(x41)
        x43 = self.conv42(x42)
        x50 = self.conv43(x43)
        light_map = x50[:,:512*3,:,:].view(x50.shape[0], 3, 512, x50.shape[2], x50.shape[3])
        confidence_map = x50[:,512*3:,:,:].view(x50.shape[0], 1, 512, x50.shape[2], x50.shape[3])

        light = light_map * confidence_map
        light = torch.mean(light.view(light.shape[0], light.shape[1], light.shape[2], -1), dim=3)

        est_source_light = light.view(light.shape[0], light.shape[1], 16, 32)

        target_light_view = target_light.contiguous().view(bz, -1, 1, 1)
        y43 = self.unconv44(target_light_view).repeat(1, 1, 48, 32)
        # print(y43.shape)
        y42 = self.unconv43(y43)
        # print(y42.shape)
        y41 = self.unconv42(torch.cat([y42, x42], dim=1))
        # print(y41.shape)
        y40 = self.unconv41(torch.cat([y41, x41], dim=1))
        # print(y40.shape)

        y31 = self.unconv40(torch.cat([y40, x40], dim=1))
        # print(y31.shape)
        y30 = self.unconv31(torch.cat([y31, x31], dim=1))
        # print(y30.shape)

        y21 = self.unconv30(torch.cat([y30, x30], dim=1))
        # print(y21.shape)
        y20 = self.unconv21(torch.cat([y21, x21], dim=1))
        # print(y20.shape)

        y11 = self.unconv20(torch.cat([y20, x20], dim=1))
        # print(y11.shape)
        y10 = self.unconv11(torch.cat([y11, x11], dim=1))
        # print(y10.shape)

        y00 = self.unconv10(torch.cat([y10, x10], dim=1))

        output = self.unconv00(torch.cat([y00, x00], dim=1))

        return output, est_source_light


class Seperate_UNet_v2(nn.Module):
    
    def __init__(self, num_group=1):
        super(Seperate_UNet_v2, self).__init__()
        self.relightnet = RelightNet_v2(num_group)
        self.shadowunet = ShadowUnet(num_group)

    def forward(self, input_human_scene, human_mask, target_light):
        rendering, est_source_light = self.relightnet(input_human_scene, target_light)
        shadow = self.shadowunet(human_mask, target_light)

        return rendering, est_source_light, shadow