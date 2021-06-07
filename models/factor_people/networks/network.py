import torch
import torch.nn.functional as F
import torch.nn as nn


class HumanNetComplete(nn.Module):
    
    def __init__(self, num_group=1, feature_channel=3):
        super(HumanNetComplete, self).__init__()

        self.conv00 = nn.Sequential(
                        nn.Conv2d(4, 29, 7, stride=1, padding=3),
                        nn.PReLU()) # 512 -> 512
        self.feature_conv0 = nn.Sequential( 
                        nn.Conv2d(32, feature_channel, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=feature_channel),
                        nn.PReLU())  
        self.conv01 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 512 -> 256

        self.conv10 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.feature_conv1 = nn.Sequential( 
                        nn.Conv2d(64, feature_channel, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=feature_channel),
                        nn.PReLU())  
        self.conv11 = nn.Sequential( 
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())  # 256 -> 128


        self.conv20 = nn.Sequential( 
                        nn.Conv2d(128, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())        
        self.feature_conv2 = nn.Sequential( 
                        nn.Conv2d(128, feature_channel, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=feature_channel),
                        nn.PReLU())    
        self.conv21 = nn.Sequential( 
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())  # 128 -> 64


        self.conv30 = nn.Sequential(
                        nn.Conv2d(256, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())
        self.feature_conv3 = nn.Sequential(
                        nn.Conv2d(256, feature_channel*4, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=feature_channel*4),
                        nn.PReLU())    
        self.conv31 = nn.Sequential(
                        nn.Conv2d(256, 512, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())  # 64 -> 32


        self.conv40 = nn.Sequential(
                        nn.Conv2d(512, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.feature_conv4 = nn.Sequential(
                        nn.Conv2d(512, feature_channel*16, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=feature_channel*16),
                        nn.PReLU())
        self.conv41 = nn.Sequential(
                        nn.Conv2d(512, 512, 3, stride=2, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) # 32->16

    def forward(self, human_mask, human_scene):

        x00 = self.conv00(torch.cat([human_scene, human_mask], dim=1))
        x00 = torch.cat([x00, human_scene], dim=1)
        hf_0 = self.feature_conv0(x00)
        x10 = self.conv01(x00)

        x11 = self.conv10(x10)
        hf_1 = self.feature_conv1(x11)
        x20 = self.conv11(x11)

        x21  = self.conv20(x20)
        hf_2 = self.feature_conv2(x21)  # 3 * 128
        x30  = self.conv21(x21)

        x31  = self.conv30(x30)
        hf_3 = self.feature_conv3(x31) # 12 * 64
        x40  = self.conv31(x31)

        x41  = self.conv40(x40)
        hf_4 = self.feature_conv4(x41) # 48 * 32
        hf   = self.conv41(x41) # 512 * 16

        return hf_0, hf_1, hf_2, hf_3, hf_4, hf


class DecoderComplete(nn.Module):

    def __init__(self, num_group=1, output_channel=3, input_channel=512, final_type='sigmoid', feature_channel=3):
        super(DecoderComplete, self).__init__()

        self.unconv5 = nn.Sequential(
                        nn.ConvTranspose2d(input_channel, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())

        self.unconv41 = nn.Sequential(
                        nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU()) # 16->32

        self.unconv40 = nn.Sequential(
                        nn.ConvTranspose2d(512+feature_channel*16, 512, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=512),
                        nn.PReLU())
        self.unconv31 = nn.Sequential(
                        nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU()) # 32 -> 64

        self.unconv30 = nn.Sequential(
                        nn.ConvTranspose2d(256+feature_channel*4, 256, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=256),
                        nn.PReLU())      
        self.unconv21 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU()) # 64 -> 128
  
        self.unconv20 = nn.Sequential(
                        nn.ConvTranspose2d(128+feature_channel, 128, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=128),
                        nn.PReLU())
        self.unconv11 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU()) # 128 -> 256

        self.unconv10 = nn.Sequential(
                        nn.ConvTranspose2d(64+feature_channel, 64, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=64),
                        nn.PReLU())
        self.unconv01 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.GroupNorm(num_groups=num_group, num_channels=32),
                        nn.PReLU()) # 256 -> 512

        if final_type == 'sigmoid':
            self.unconv00 = nn.Sequential(
                            nn.ConvTranspose2d(32+feature_channel, output_channel, 3, stride=1, padding=1),
                            nn.Sigmoid())
        elif final_type == 'relu':
            self.unconv00 = nn.Sequential(
                            nn.ConvTranspose2d(32+feature_channel, output_channel, 3, stride=1, padding=1),
                            nn.ReLU())

    def forward(self, hf_0, hf_1, hf_2, hf_3, hf_4, middle_f):
        
        x = self.unconv5(middle_f)
        x = self.unconv41(x)

        x = self.unconv40(torch.cat([x, hf_4], dim=1))
        x = self.unconv31(x)

        x = self.unconv30(torch.cat([x, hf_3], dim=1))
        x = self.unconv21(x)

        x = self.unconv20(torch.cat([x, hf_2], dim=1))
        x = self.unconv11(x)

        x = self.unconv10(torch.cat([x, hf_1], dim=1))
        x = self.unconv01(x)

        output = self.unconv00(torch.cat([x, hf_0], dim=1))

        return output


class SepNetComplete_Shading_SHLight(nn.Module):

    def __init__(self, f_channel=3, use_intensity=1):
        super(SepNetComplete_Shading_SHLight, self).__init__()

        self.humanNet = HumanNetComplete(feature_channel=f_channel, num_group=16)

        self.use_intensity = use_intensity

        if use_intensity != 0:
            self.shadingDecoder = DecoderComplete(input_channel=512+27+512+3, final_type='relu',feature_channel=f_channel, num_group=16)
        else:
            self.shadingDecoder = DecoderComplete(input_channel=512+27+512, final_type='relu',feature_channel=f_channel, num_group=16)

    def extractFeatures(self, rendering, mask):

        hf_list = self.humanNet(mask, rendering)

        return hf_list

    def decodeShading(self, hf_list, shlight, sun_map, sun_intensity, mask):


        shlight_f = shlight.reshape(shlight.shape[0], -1, 1, 1).repeat(1, 1, 24, 16)
        sun_map_f = sun_map.reshape(sun_map.shape[0], -1, 1, 1).repeat(1, 1, 24, 16)
        sun_intensity_f = sun_intensity.reshape(sun_intensity.shape[0], -1, 1, 1).repeat(1, 1, 24, 16)

        if self.use_intensity != 0:
            hf_mid_w_light = torch.cat([hf_list[5], shlight_f, sun_map_f, sun_intensity_f], dim=1)
        else:
            hf_mid_w_light = torch.cat([hf_list[5], shlight_f, sun_map_f], dim=1)


        shading = self.shadingDecoder(hf_list[0], hf_list[1], hf_list[2], hf_list[3], hf_list[4], hf_mid_w_light)

        return shading * mask

    def forward(self, rendering, mask, shlight, sun_map, sun_intensity):
        est_hf = self.extractFeatures(rendering, mask)
        est_shading = self.decodeShading(est_hf, shlight, sun_map, sun_intensity, mask)

        return est_shading


class SepNetComplete_Shading(nn.Module):

    def __init__(self, f_channel=16):
        super(SepNetComplete_Shading, self).__init__()

        self.humanNet = HumanNetComplete(feature_channel=f_channel, num_group=16)

        self.shadingDecoder = DecoderComplete(input_channel=512*4, final_type='relu',feature_channel=f_channel, num_group=16)

    def extractFeatures(self, rendering, mask):

        hf_list = self.humanNet(mask, rendering)

        return hf_list

    def decodeShading(self, hf_list, light, mask):

        light_f = light.reshape(light.shape[0], -1, 1, 1).repeat(1, 1, 24, 16)

        hf_mid_w_light = torch.cat([hf_list[5], light_f], dim=1)

        shading = self.shadingDecoder(hf_list[0], hf_list[1], hf_list[2], hf_list[3], hf_list[4], hf_mid_w_light)

        return shading * mask

    def forward(self, rendering, mask, light):
        est_hf = self.extractFeatures(rendering, mask)
        est_shading = self.decodeShading(est_hf, light, mask)

        return est_shading


class SepNetComplete_HumanIntrinsic(nn.Module):

    def __init__(self, f_channel=3, output_channel=3, activate_f='relu'):
        super(SepNetComplete_HumanIntrinsic, self).__init__()

        self.humanNet = HumanNetComplete(feature_channel=f_channel, num_group=16)

        self.Decoder = DecoderComplete(input_channel=512, output_channel=output_channel, final_type=activate_f, feature_channel=f_channel, num_group=16)

    def extractFeatures(self, rendering, mask):

        hf_list = self.humanNet(mask, rendering)

        return hf_list

    def decode(self, hf_list, mask):

        factor = self.Decoder(hf_list[0], hf_list[1], hf_list[2], hf_list[3], hf_list[4], hf_list[5])

        return factor * mask

    def forward(self, rendering, mask):
        est_hf = self.extractFeatures(rendering, mask)
        est_factor = self.decode(est_hf, mask)

        return est_factor

class Unet(nn.Module):

    def __init__(self, num_group=16):
        super(Unet, self).__init__()

        self.conv0 = nn.Sequential(
                    nn.Conv2d(4, 29, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
                    nn.ReLU()) # 512 -> 512


    def forward(self, rendering, mask):
        input = torch.cat([rendering, mask], dim=1)

        x0 = self.conv0(input)
        x0 = torch.cat([x0, rendering], dim=1)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        ux4 = self.unconv5(x6)
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output

class Unet_w_light(nn.Module):

    def __init__(self, num_group=16):
        super(Unet_w_light, self).__init__()

        self.conv0 = nn.Sequential(
                    nn.Conv2d(4, 29, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.ConvTranspose2d(2048, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
                    nn.ReLU()) # 512 -> 512


    def forward(self, rendering, mask, light):
        input = torch.cat([rendering, mask], dim=1)

        x0 = self.conv0(input)
        x0 = torch.cat([x0, rendering], dim=1)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        light = light.reshape(light.shape[0], -1, 1, 1).repeat(1, 1, 48, 32)

        ux4 = self.unconv5(torch.cat([x6, light], dim=1))
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output

class Unet_Refine(nn.Module):

    def __init__(self, num_group=16, input_channel=4):
        super(Unet_Refine, self).__init__()

        self.conv0 = nn.Sequential(
                    nn.Conv2d(input_channel, 32, 3, stride=1, padding=1),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
                    nn.ReLU()) # 512 -> 512


    def forward(self, input):

        x0 = self.conv0(input)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        ux4 = self.unconv5(x6)
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output


class Unet_Blurpooling(nn.Module):

    def __init__(self, num_group=16):
        super(Unet_Blurpooling, self).__init__()

        bk = torch.FloatTensor([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]).reshape(1, 1, 3, 3)
        bk = bk / torch.sum(bk)
        self.kernel_list = []

        for i in range(4):
            c = 64 * (2 ** i)
            kernel_tensor = torch.zeros((c, c, 3, 3))
            kernel_tensor[range(c), range(c), :, :] = bk
            self.kernel_list.append(kernel_tensor)


        self.conv0 = nn.Sequential(
                    nn.Conv2d(4, 29, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.Conv2d(1024, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.Conv2d(512, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.Conv2d(256, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.Conv2d(64, 3, 3, stride=1, padding=1),
                    nn.ReLU()) # 512 -> 512

    def blurpooling(self, x, i):
        return F.conv2d(x, self.kernel_list[i], stride=2, padding = 1)

    def cuda_kernels(self):
        for i in range(len(self.kernel_list)):
            self.kernel_list[i] = self.kernel_list[i].cuda()

    def forward(self, rendering, mask):
        input = torch.cat([rendering, mask], dim=1)

        x0 = self.conv0(input)
        x0 = torch.cat([x0, rendering], dim=1)

        x1 = self.conv1(x0)
        x1 = self.blurpooling(x1, 0)
        x2 = self.conv2(x1)
        x2 = self.blurpooling(x2, 1)
        x3 = self.conv3(x2)
        x3 = self.blurpooling(x3, 2)
        x4 = self.conv4(x3)
        x4 = self.blurpooling(x4, 3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        ux4 = self.unconv5(x6)
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux3 = F.interpolate(ux3, scale_factor=2, mode='bilinear', align_corners=False)
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux2 = F.interpolate(ux2, scale_factor=2, mode='bilinear', align_corners=False)
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux1 = F.interpolate(ux1, scale_factor=2, mode='bilinear', align_corners=False)
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))
        ux0 = F.interpolate(ux0, scale_factor=2, mode='bilinear', align_corners=False)

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output

class Unet_Blurpooling_General(nn.Module):

    def __init__(self, num_group=16, input_channel=4, output_channel=3, allow_negative=False):
        super(Unet_Blurpooling_General, self).__init__()

        bk = torch.FloatTensor([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]).reshape(1, 1, 3, 3)
        bk = bk / torch.sum(bk)
        self.kernel_list = []

        for i in range(4):
            c = 64 * (2 ** i)
            kernel_tensor = torch.zeros((c, c, 3, 3))
            kernel_tensor[range(c), range(c), :, :] = bk
            self.kernel_list.append(kernel_tensor)


        self.conv0 = nn.Sequential(
                    nn.Conv2d(input_channel, 32, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.Conv2d(1024, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.Conv2d(512, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.Conv2d(256, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        if allow_negative:
            self.unconv0 = nn.Sequential(
                    nn.Conv2d(64, output_channel, 3, stride=1, padding=1)
                    ) # 512 -> 512
        else:
            self.unconv0 = nn.Sequential(
                    nn.Conv2d(64, output_channel, 3, stride=1, padding=1),
                    nn.ReLU()
                    ) # 512 -> 512

    def blurpooling(self, x, i):
        return F.conv2d(x, self.kernel_list[i], stride=2, padding = 1)

    def cuda_kernels(self):
        for i in range(len(self.kernel_list)):
            self.kernel_list[i] = self.kernel_list[i].cuda()

    def forward(self, input):

        x0 = self.conv0(input)

        x1 = self.conv1(x0)
        x1 = self.blurpooling(x1, 0)
        x2 = self.conv2(x1)
        x2 = self.blurpooling(x2, 1)
        x3 = self.conv3(x2)
        x3 = self.blurpooling(x3, 2)
        x4 = self.conv4(x3)
        x4 = self.blurpooling(x4, 3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        ux4 = self.unconv5(x6)
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux3 = F.interpolate(ux3, scale_factor=2, mode='bilinear', align_corners=False)
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux2 = F.interpolate(ux2, scale_factor=2, mode='bilinear', align_corners=False)
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux1 = F.interpolate(ux1, scale_factor=2, mode='bilinear', align_corners=False)
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))
        ux0 = F.interpolate(ux0, scale_factor=2, mode='bilinear', align_corners=False)

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output

class Unet_Blurpooling_General_Light(nn.Module):

    def __init__(self, num_group=16, input_channel=4):
        super(Unet_Blurpooling_General_Light, self).__init__()

        bk = torch.FloatTensor([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]).reshape(1, 1, 3, 3)
        bk = bk / torch.sum(bk)
        self.kernel_list = []

        for i in range(4):
            c = 64 * (2 ** i)
            kernel_tensor = torch.zeros((c, c, 3, 3))
            kernel_tensor[range(c), range(c), :, :] = bk
            self.kernel_list.append(kernel_tensor)


        self.conv0 = nn.Sequential(
                    nn.Conv2d(input_channel, 32, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.Conv2d(2048, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.Conv2d(1024, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.Conv2d(512, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.Conv2d(256, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.Conv2d(64, 3, 3, stride=1, padding=1),
                    nn.ReLU()) # 512 -> 512

    def blurpooling(self, x, i):
        return F.conv2d(x, self.kernel_list[i], stride=2, padding = 1)

    def cuda_kernels(self):
        for i in range(len(self.kernel_list)):
            self.kernel_list[i] = self.kernel_list[i].cuda()

    def forward(self, input, light):

        x0 = self.conv0(input)

        x1 = self.conv1(x0)
        x1 = self.blurpooling(x1, 0)
        x2 = self.conv2(x1)
        x2 = self.blurpooling(x2, 1)
        x3 = self.conv3(x2)
        x3 = self.blurpooling(x3, 2)
        x4 = self.conv4(x3)
        x4 = self.blurpooling(x4, 3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        light = light.reshape(light.shape[0], -1, 1, 1).repeat(1, 1, 48, 32)

        ux4 = self.unconv5(torch.cat([x6, light], dim=1))
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux3 = F.interpolate(ux3, scale_factor=2, mode='bilinear', align_corners=False)
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux2 = F.interpolate(ux2, scale_factor=2, mode='bilinear', align_corners=False)
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux1 = F.interpolate(ux1, scale_factor=2, mode='bilinear', align_corners=False)
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))
        ux0 = F.interpolate(ux0, scale_factor=2, mode='bilinear', align_corners=False)

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return output



class Unet_Blurpooling_Shadow(nn.Module):

    def __init__(self, num_group=16):
        super(Unet_Blurpooling_Shadow, self).__init__()

        bk = torch.FloatTensor([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]).reshape(1, 1, 3, 3)
        bk = bk / torch.sum(bk)
        self.kernel_list = []

        for i in range(4):
            c = 64 * (2 ** i)
            kernel_tensor = torch.zeros((c, c, 3, 3))
            kernel_tensor[range(c), range(c), :, :] = bk
            self.kernel_list.append(kernel_tensor)


        self.conv0 = nn.Sequential(
                    nn.Conv2d(4, 32, 7, stride=1, padding=3),
                    nn.LeakyReLU()) # 512 -> 512
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 512 -> 256

        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 256 -> 128

        self.conv3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 128 -> 64

        self.conv4 = nn.Sequential(
                    nn.Conv2d(256, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 64 -> 32

        self.conv5 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.conv6 = nn.Sequential(
                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv5 = nn.Sequential(
                    nn.Conv2d(2048, 512, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=512),
                    nn.LeakyReLU()) # 32

        self.unconv4 = nn.Sequential(
                    nn.Conv2d(1024, 256, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=256),
                    nn.LeakyReLU()) # 32 -> 64
        
        self.unconv3 = nn.Sequential(
                    nn.Conv2d(512, 128, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=128),
                    nn.LeakyReLU()) # 64 -> 128

        self.unconv2 = nn.Sequential(
                    nn.Conv2d(256, 64, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=64),
                    nn.LeakyReLU()) # 128 -> 256

        self.unconv1 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=num_group, num_channels=32),
                    nn.LeakyReLU()) # 256 -> 512

        self.unconv0 = nn.Sequential(
                    nn.Conv2d(64, 1, 3, stride=1, padding=1),
                    nn.Tanh()) # 512 -> 512

    def blurpooling(self, x, i):
        return F.conv2d(x, self.kernel_list[i], stride=2, padding = 1)

    def cuda_kernels(self):
        for i in range(len(self.kernel_list)):
            self.kernel_list[i] = self.kernel_list[i].cuda()

    def forward(self, input, light):

        x0 = self.conv0(input)

        x1 = self.conv1(x0)
        x1 = self.blurpooling(x1, 0)
        x2 = self.conv2(x1)
        x2 = self.blurpooling(x2, 1)
        x3 = self.conv3(x2)
        x3 = self.blurpooling(x3, 2)
        x4 = self.conv4(x3)
        x4 = self.blurpooling(x4, 3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        light = light.reshape(light.shape[0], -1, 1, 1).repeat(1, 1, 48, 32)

        ux4 = self.unconv5(torch.cat([x6, light], dim=1))
        ux3 = self.unconv4(torch.cat([ux4, x4], dim=1))
        ux3 = F.interpolate(ux3, scale_factor=2, mode='bilinear', align_corners=False)
        ux2 = self.unconv3(torch.cat([ux3, x3], dim=1))
        ux2 = F.interpolate(ux2, scale_factor=2, mode='bilinear', align_corners=False)
        ux1 = self.unconv2(torch.cat([ux2, x2], dim=1))
        ux1 = F.interpolate(ux1, scale_factor=2, mode='bilinear', align_corners=False)
        ux0 = self.unconv1(torch.cat([ux1, x1], dim=1))
        ux0 = F.interpolate(ux0, scale_factor=2, mode='bilinear', align_corners=False)

        output = self.unconv0(torch.cat([ux0, x0], dim=1))

        return 0.5 * (output + 1.)