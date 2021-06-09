import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k_size=3, s=1, num_group=1):
        super(ResBlock, self).__init__()

        self.c0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k_size, stride=s, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=out_channel),
            nn.PReLU(),
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, k_size, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=out_channel),
        )

        if in_channel != out_channel or s != 1:
            self.c = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=s),
                nn.GroupNorm(num_groups=num_group, num_channels=out_channel),
            )

    def forward(self, x):
        h = self.c0(x)
        h = self.c1(h)

        if x.shape[1] != h.shape[1] or x.shape[2] != h.shape[2]:
            x = self.c(x)

        return x + h


class LightNet_from_ft(nn.Module):
    def __init__(self, input_channels, inter_channels=10):
        super(LightNet_from_ft, self).__init__()

        self.common_net = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(128, inter_channels, kernel_size=1),
            nn.PReLU(),
            nn.Flatten(),
        )
        self.ground_map_net = nn.Linear(inter_channels * 16 * 24, 3 * 16 * 32)
        self.sun_map_net = nn.Linear(inter_channels * 16 * 24, 3 * 3 * 9)
        self.sun_intensity_net = nn.Linear(inter_channels * 16 * 24, 3)

    def forward(self, x):
        ft = self.common_net(x)
        ground_map = self.ground_map_net(ft)
        sun_map = self.sun_map_net(ft)
        sun_intensity = self.sun_intensity_net(ft)
        return (
            ground_map.view(-1, 3, 16, 32),
            sun_map.view(-1, 3, 3, 9),
            sun_intensity.view(-1, 3),
        )


class LightDirectNet(nn.Module):
    def __init__(self, num_group=1):
        super(LightDirectNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2, num_group), nn.PReLU())  # 64

        self.conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, num_group), nn.PReLU())

        self.conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, num_group), nn.PReLU())

        self.conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.sun_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, num_group), nn.PReLU()
        )  # 8
        self.sun_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.sun_dense = nn.Sequential(nn.Linear(4 * 6 * 512, 512 * 3), nn.Sigmoid())

        self.avgpooling = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, img, mask):
        input = torch.cat([img, mask], dim=1)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.conv8(x)

        xx = self.sun_conv1(x)
        xx = self.sun_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        sun = self.sun_dense(xx)
        sun = sun.reshape(sun.shape[0], 3, 16, 32)

        return sun


class LightNet_Two_Part_sh(nn.Module):
    def __init__(self, num_group=1):
        super(LightNet_Two_Part_sh, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2, num_group), nn.PReLU())  # 64

        self.conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, num_group), nn.PReLU())

        self.conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, num_group), nn.PReLU())

        self.conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.ground_conv1 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16
        self.ground_conv2 = nn.Sequential(
            ResBlock(512, 512, 3, 1, num_group), nn.PReLU()
        )
        self.ground_para_dense1 = nn.Sequential(
            nn.Linear(8 * 12 * 512, 512), nn.PReLU()
        )
        self.ground_para_dense2 = nn.Sequential(nn.Linear(512, 27))

        self.sun_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, num_group), nn.PReLU()
        )  # 8
        self.sun_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.sun_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.sun_para_dense2 = nn.Sequential(nn.Linear(512, 1), nn.ReLU())
        self.sun_dense = nn.Sequential(nn.Linear(4 * 6 * 512, 512 * 3), nn.Sigmoid())

        self.avgpooling = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, img, mask):
        input = torch.cat([img, mask], dim=1)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        xx = self.ground_conv1(x)
        xx = self.ground_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        ground_shlight = self.ground_para_dense2(self.ground_para_dense1(xx))
        ground_shlight = ground_shlight.reshape(ground_shlight.shape[0], 3, 9)

        x = self.conv7(x)
        x = self.conv8(x)

        xx = self.sun_conv1(x)
        xx = self.sun_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        sun_para = self.sun_para_dense2(self.sun_para_dense1(xx))
        sun_para = sun_para.reshape(sun_para.shape[0], 1, 1, 1)
        sun = self.sun_dense(xx)
        sun = sun.reshape(sun.shape[0], 3, 16, 32)

        return ground_shlight, sun, sun_para.reshape(sun_para.shape[0], 1, 1, 1)


class LightNet_Two_Part_sh_single(nn.Module):
    def __init__(self, num_group=1, input_channel=4):
        super(LightNet_Two_Part_sh_single, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2, num_group), nn.PReLU())  # 64

        self.conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, num_group), nn.PReLU())

        self.conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, num_group), nn.PReLU())

        self.conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.ground_conv1 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16
        self.ground_conv2 = nn.Sequential(
            ResBlock(512, 512, 3, 1, num_group), nn.PReLU()
        )
        self.ground_para_dense1 = nn.Sequential(
            nn.Linear(8 * 12 * 512, 512), nn.PReLU()
        )
        self.ground_para_dense2 = nn.Sequential(nn.Linear(512, 27))

        self.sun_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, num_group), nn.PReLU()
        )  # 8
        self.sun_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.sun_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.sun_para_dense2 = nn.Sequential(nn.Linear(512 + 27, 3), nn.ReLU())
        self.sun_para_dense3 = nn.Sequential(nn.Linear(512 + 27, 1), nn.Sigmoid())
        self.sun_dense = nn.Sequential(
            nn.Linear(4 * 6 * 512 + 512, 512), nn.LogSoftmax(dim=1)
        )

        self.avgpooling = nn.AvgPool2d(3, stride=2, padding=1)

    def getEnvLightConf(self, shlight):
        light = shlight.reshape(shlight.shape[0], 3, 9, 1, 1)
        envlight = torch.sum(light * env_transport_tensor, dim=2)
        # print(envlight)

        Intensity = torch.sum(torch.clamp(envlight, 0.0, float("inf")), dim=1).reshape(
            envlight.shape[0], -1
        )

        return Intensity / torch.sum(Intensity, dim=1, keepdim=True)

    def forward(self, human, mask, gth_ground_shlight=None, crop_mask=None):
        if crop_mask is None:
            input = torch.cat([human, mask], dim=1)
        else:
            input = torch.cat([human, mask, crop_mask], dim=1)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        xx = self.ground_conv1(x)
        xx = self.ground_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        ground_shlight = self.ground_para_dense2(self.ground_para_dense1(xx))

        if gth_ground_shlight is None:
            env_conf = self.getEnvLightConf(ground_shlight)
        else:
            env_conf = self.getEnvLightConf(gth_ground_shlight)

        x = self.conv7(x)
        x = self.conv8(x)

        xx = self.sun_conv1(x)
        xx = self.sun_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        dense_feature = self.sun_para_dense1(xx)

        if gth_ground_shlight is None:
            sun_para = self.sun_para_dense2(
                torch.cat([dense_feature, ground_shlight], dim=1)
            )
        else:
            sun_para = self.sun_para_dense2(
                torch.cat(
                    [
                        dense_feature,
                        gth_ground_shlight.reshape(gth_ground_shlight.shape[0], -1),
                    ],
                    dim=1,
                )
            )

        if gth_ground_shlight is None:
            sun_threshold = self.sun_para_dense3(
                torch.cat([dense_feature, ground_shlight], dim=1)
            )
        else:
            sun_threshold = self.sun_para_dense3(
                torch.cat(
                    [
                        dense_feature,
                        gth_ground_shlight.reshape(gth_ground_shlight.shape[0], -1),
                    ],
                    dim=1,
                )
            )

        sun_para = sun_para.reshape(sun_para.shape[0], 3)
        sun = self.sun_dense(torch.cat([xx, env_conf], dim=1))
        sun = sun.reshape(sun.shape[0], 1, 16, 32)

        ground_shlight = ground_shlight.reshape(ground_shlight.shape[0], 3, 9)

        est_position = torch.sum(
            (coord_tensor * torch.exp(sun)).reshape(sun.shape[0], 2, -1), axis=2
        )

        return (
            ground_shlight,
            sun,
            sun_para,
            env_conf.reshape(env_conf.shape[0], 1, 16, 32),
            est_position,
            sun_threshold.reshape(-1),
        )


def getEnvTransport():
    hdr_width = 32
    hdr_height = 16

    angles = np.zeros((hdr_height, hdr_width, 2))

    for i in range(hdr_height):
        for j in range(hdr_width):
            angles[i, j] = [
                i / hdr_height * np.pi,
                0.5 * np.pi - j / hdr_width * np.pi * 2,
            ]

    normal = np.zeros((hdr_height, hdr_width, 3))
    for i in range(hdr_height):
        for j in range(hdr_width):
            theta = angles[i, j, 0]
            phi = angles[i, j, 1]
            normal[i, j] = [
                np.cos(phi) * np.sin(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(theta),
            ]
            normal[i, j] = normal[i, j] / np.linalg.norm(normal[i, j])

    return getTransportMap(normal)


def getShading(transport, lighting):
    Light = lighting.reshape(1, 1, lighting.shape[0], lighting.shape[1])
    transport_map = np.expand_dims(transport, axis=3)
    shading = np.sum(Light * transport_map, axis=2)
    return shading


def getTransportMap(normal):
    img_shape = normal.shape[0:2]
    transport_map = np.zeros((img_shape[0], img_shape[1], 9), dtype=float)
    transport_map[:, :, 0] = 1.0
    transport_map[:, :, 1] = normal[:, :, 0]
    transport_map[:, :, 2] = normal[:, :, 1]
    transport_map[:, :, 3] = normal[:, :, 2]
    transport_map[:, :, 4] = 3 * normal[:, :, 2] * normal[:, :, 2] - 1.0
    transport_map[:, :, 5] = normal[:, :, 0] * normal[:, :, 1]
    transport_map[:, :, 6] = normal[:, :, 0] * normal[:, :, 2]
    transport_map[:, :, 7] = normal[:, :, 1] * normal[:, :, 2]
    transport_map[:, :, 8] = (
        normal[:, :, 0] * normal[:, :, 0] - normal[:, :, 1] * normal[:, :, 1]
    )

    return transport_map


env_transport_tensor = getEnvTransport()
env_transport_tensor = torch.FloatTensor(
    np.expand_dims(np.expand_dims(env_transport_tensor.transpose(2, 0, 1), 0), 0)
).cuda()
coord_tensor = np.zeros((1, 2, 16, 32))
for r in range(16):
    for c in range(32):
        coord_tensor[0, :, r, c] = [r, c]
coord_tensor = torch.FloatTensor(coord_tensor).cuda()


class LightNet_Hybrid(nn.Module):
    def __init__(self, num_group=1, input_channel=4):
        super(LightNet_Hybrid, self).__init__()

        self.human_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.human_conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.human_conv3 = nn.Sequential(
            ResBlock(64, 128, 3, 2, num_group), nn.PReLU()
        )  # 64

        self.human_conv4 = nn.Sequential(
            ResBlock(128, 128, 3, 1, num_group), nn.PReLU()
        )

        self.human_conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.human_conv6 = nn.Sequential(
            ResBlock(256, 256, 3, 1, num_group), nn.PReLU()
        )

        self.human_conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.human_conv8 = nn.Sequential(
            ResBlock(512, 512, 3, 1, num_group), nn.PReLU()
        )

        self.human_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.bg_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128
        self.bg_conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.bg_conv3 = nn.Sequential(
            ResBlock(64, 128, 3, 2, num_group), nn.PReLU()
        )  # 64

        self.bg_conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, num_group), nn.PReLU())

        self.bg_conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.bg_conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, num_group), nn.PReLU())

        self.bg_conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.bg_conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=num_group, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, num_group), nn.PReLU())

        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2, num_group), nn.PReLU())  # 64

        self.conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, num_group), nn.PReLU())

        self.conv5 = nn.Sequential(
            ResBlock(128, 256, 3, 2, num_group), nn.PReLU()
        )  # 32

        self.conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, num_group), nn.PReLU())

        self.conv7 = nn.Sequential(
            ResBlock(256, 512, 3, 2, num_group), nn.PReLU()
        )  # 16

        self.conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.ground_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, num_group), nn.PReLU()
        )  # 16
        self.ground_conv2 = nn.Sequential(
            ResBlock(512, 512, 3, 1, num_group), nn.PReLU()
        )
        self.ground_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.ground_para_dense2 = nn.Sequential(nn.Linear(512, 27))

        self.sun_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, num_group), nn.PReLU()
        )  # 8
        self.sun_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, num_group), nn.PReLU())

        self.sun_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.sun_para_dense2 = nn.Sequential(nn.Linear(512 + 27, 3), nn.ReLU())
        self.sun_dense = nn.Sequential(
            nn.Linear(4 * 6 * 512 + 512, 512), nn.LogSoftmax(dim=1)
        )

        self.sun_map_threshold_dense = nn.Sequential(
            nn.Linear(3 * 512, 3), nn.LogSoftmax(dim=1)
        )
        self.sun_intensity_threshold_dense = nn.Sequential(
            nn.Linear(3 * 512, 3), nn.LogSoftmax(dim=1)
        )
        self.ground_threshold_dense = nn.Sequential(
            nn.Linear(3 * 512, 3), nn.LogSoftmax(dim=1)
        )

        self.avgpooling = nn.AdaptiveAvgPool2d((4, 6))
        self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))

    def getEnvLightConf(self, shlight):
        light = shlight.reshape(shlight.shape[0], 3, 9, 1, 1)
        envlight = torch.sum(light * env_transport_tensor, dim=2)

        Intensity = torch.sum(torch.clamp(envlight, 0.0, float("inf")), dim=1).reshape(
            envlight.shape[0], -1
        )

        return Intensity / torch.sum(Intensity, dim=1, keepdim=True)

    def encode_human_bg(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.conv8(x)

        return x

    def encode_human(self, input):
        x = self.human_conv1(input)
        x = self.human_conv2(x)
        x = self.human_conv3(x)
        x = self.human_conv4(x)

        x = self.human_conv5(x)
        x = self.human_conv6(x)

        x = self.human_conv7(x)
        x = self.human_conv8(x)

        return x

    def encode_bg(self, input):
        x = self.bg_conv1(input)
        x = self.bg_conv2(x)
        x = self.bg_conv3(x)
        x = self.bg_conv4(x)

        x = self.bg_conv5(x)
        x = self.bg_conv6(x)

        x = self.bg_conv7(x)
        x = self.bg_conv8(x)

        return x

    def decode(self, x, gth_ground_shlight=None):
        xx = self.ground_conv1(x)
        xx = self.ground_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        ground_shlight = self.ground_para_dense2(self.ground_para_dense1(xx))

        if gth_ground_shlight is None:
            env_conf = self.getEnvLightConf(ground_shlight)
        else:
            env_conf = self.getEnvLightConf(gth_ground_shlight)

        xx = self.sun_conv1(x)
        xx = self.sun_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        dense_feature = self.sun_para_dense1(xx)

        if gth_ground_shlight is None:
            sun_para = self.sun_para_dense2(
                torch.cat([dense_feature, ground_shlight], dim=1)
            )
        else:
            sun_para = self.sun_para_dense2(
                torch.cat(
                    [
                        dense_feature,
                        gth_ground_shlight.reshape(gth_ground_shlight.shape[0], -1),
                    ],
                    dim=1,
                )
            )

        sun_para = sun_para.reshape(sun_para.shape[0], 3)
        sun = self.sun_dense(torch.cat([xx, env_conf], dim=1))
        sun = sun.reshape(sun.shape[0], 1, 16, 32)

        ground_shlight = ground_shlight.reshape(ground_shlight.shape[0], 3, 9)

        return ground_shlight, sun, sun_para

    def forward(self, human, mask, gth_ground_shlight=None, crop_mask=None):
        print(self.human_conv1[0].weight.grad)
        if crop_mask is None:
            all_f = self.encode_human_bg(torch.cat([human, mask], dim=1))
            human_f = self.encode_human(torch.cat([human * mask, mask], dim=1))
            bg_f = self.encode_bg(torch.cat([human * (1.0 - mask), mask], dim=1))
        else:
            all_f = self.encode_human_bg(torch.cat([human, mask, crop_mask], dim=1))
            human_f = self.encode_human(
                torch.cat([human * mask, mask, crop_mask], dim=1)
            )
            bg_f = self.encode_bg(
                torch.cat([human * (1.0 - mask), mask, crop_mask], dim=1)
            )

        ground_shlight, sun, sun_para = self.decode(all_f, gth_ground_shlight)
        human_ground_shlight, human_sun, human_sun_para = self.decode(
            human_f, gth_ground_shlight
        )
        bg_ground_shlight, bg_sun, bg_sun_para = self.decode(bg_f, gth_ground_shlight)

        all_f = self.maxpooling(all_f).reshape(all_f.shape[0], -1)
        human_f = self.maxpooling(human_f).reshape(human_f.shape[0], -1)
        bg_f = self.maxpooling(bg_f).reshape(bg_f.shape[0], -1)
        ground_threshold = self.ground_threshold_dense(
            torch.cat([all_f, human_f, bg_f], dim=1)
        )
        sun_threshold = self.sun_map_threshold_dense(
            torch.cat([all_f, human_f, bg_f], dim=1)
        )
        sun_intensity_threshold = self.sun_intensity_threshold_dense(
            torch.cat([all_f, human_f, bg_f], dim=1)
        )

        return (
            ground_shlight,
            sun,
            sun_para,
            human_ground_shlight,
            human_sun,
            human_sun_para,
            bg_ground_shlight,
            bg_sun,
            bg_sun_para,
            ground_threshold,
            sun_threshold,
            sun_intensity_threshold,
        )


class AlexNet_LM(nn.Module):
    def __init__(self, num_group=16):
        super(AlexNet_LM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4),
            nn.GroupNorm(num_groups=num_group, num_channels=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2
            ),
            nn.GroupNorm(num_groups=num_group, num_channels=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.GroupNorm(num_groups=num_group, num_channels=384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=384),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((12, 8)),
        )

        # 需要针对上一层改变view
        self.layer06 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer07 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer08 = nn.Linear(in_features=4096, out_features=9)

        self.layer16 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer17 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer18 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256), nn.LogSoftmax(dim=1)
        )

    def forward(self, human, mask):
        x = torch.cat([human, mask], dim=1)
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 12 * 8 * 256)
        paras = self.layer08(self.layer07(self.layer06(x)))
        sun_position = self.layer18(self.layer17(self.layer16(x))).reshape(-1, 1, 8, 32)

        beta = paras[:, 0:1]
        kai = paras[:, 1:2]
        w_sun = paras[:, 2:5]
        t = paras[:, 5:6]
        w_sky = paras[:, 6:]

        return beta, kai, w_sun, t, w_sky, sun_position


class AlexNet_Single_SH(nn.Module):
    def __init__(self, num_group=16, input_channel=4):
        super(AlexNet_Single_SH, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel, out_channels=96, kernel_size=11, stride=4
            ),
            nn.GroupNorm(num_groups=num_group, num_channels=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2
            ),
            nn.GroupNorm(num_groups=num_group, num_channels=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.GroupNorm(num_groups=num_group, num_channels=384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=384),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_group, num_channels=256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((12, 8)),
        )

        # 需要针对上一层改变view
        self.layer06 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer07 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer08 = nn.Linear(in_features=4096, out_features=4)

        self.layer16 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer17 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer18 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=512), nn.LogSoftmax(dim=1)
        )

        self.layer26 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer27 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer28 = nn.Sequential(nn.Linear(in_features=4096, out_features=27),)

    def forward(self, human, mask, nothing, crop_mask=None):
        if crop_mask is None:
            x = torch.cat([human, mask], dim=1)
        else:
            x = torch.cat([human, mask, crop_mask], dim=1)
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 12 * 8 * 256)
        paras = self.layer08(self.layer07(self.layer06(x)))
        sun = self.layer18(self.layer17(self.layer16(x))).reshape(-1, 1, 16, 32)
        ground_shlight = self.layer28(self.layer27(self.layer26(x))).reshape(-1, 3, 9)

        sun_para = F.relu(paras[:, 0:3])
        sun_threshold = torch.sigmoid(paras[:, 3:])

        est_position = torch.sum(
            (coord_tensor * torch.exp(sun)).reshape(sun.shape[0], 2, -1), axis=2
        )

        return (
            ground_shlight,
            sun,
            sun_para,
            None,
            est_position,
            sun_threshold.reshape(-1),
        )


class LightNet(nn.Module):
    def __init__(self, input_channel=4):
        super(LightNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128

        self.conv2 = nn.Sequential(ResBlock(64, 64, 3, 1, 4), nn.PReLU())

        self.conv3 = nn.Sequential(ResBlock(64, 128, 3, 2, 8), nn.PReLU())  # 64

        self.conv4 = nn.Sequential(ResBlock(128, 128, 3, 1, 8), nn.PReLU())

        self.conv5 = nn.Sequential(ResBlock(128, 256, 3, 2, 16), nn.PReLU())  # 32

        self.conv6 = nn.Sequential(ResBlock(256, 256, 3, 1, 16), nn.PReLU())

        self.conv7 = nn.Sequential(ResBlock(256, 512, 3, 2, 16), nn.PReLU())  # 16

        self.conv8 = nn.Sequential(ResBlock(512, 512, 3, 1, 16), nn.PReLU())

        self.ground_conv1 = nn.Sequential(
            ResBlock(512, 512, 3, 2, 16), nn.PReLU()
        )  # 16
        self.ground_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, 16), nn.PReLU())
        self.ground_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.ground_para_dense2 = nn.Sequential(nn.Linear(512, 27))

        self.sun_conv1 = nn.Sequential(ResBlock(512, 512, 3, 2, 16), nn.PReLU())  # 8
        self.sun_conv2 = nn.Sequential(ResBlock(512, 512, 3, 1, 16), nn.PReLU())

        self.sun_para_dense1 = nn.Sequential(nn.Linear(4 * 6 * 512, 512), nn.PReLU())
        self.sun_para_dense2 = nn.Sequential(nn.Linear(512 + 27, 3), nn.ReLU())
        self.sun_dense = nn.Sequential(
            nn.Linear(4 * 6 * 512 + 512, 512), nn.LogSoftmax(dim=1)
        )

        self.avgpooling = nn.AdaptiveAvgPool2d((4, 6))
        self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))

    def getEnvLightConf(self, shlight):
        light = shlight.reshape(shlight.shape[0], 3, 9, 1, 1)
        envlight = torch.sum(light * env_transport_tensor, dim=2)

        Intensity = torch.sum(torch.clamp(envlight, 0.0, float("inf")), dim=1).reshape(
            envlight.shape[0], -1
        )

        return Intensity / (1e-16 + torch.sum(Intensity, dim=1, keepdim=True))

    def encode(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.conv8(x)

        return x

    def decode(self, x, gth_ground_shlight=None):
        xx = self.ground_conv1(x)
        xx = self.ground_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        ground_shlight = self.ground_para_dense2(self.ground_para_dense1(xx))

        if gth_ground_shlight is None:
            env_conf = self.getEnvLightConf(ground_shlight)
        else:
            env_conf = self.getEnvLightConf(gth_ground_shlight)

        xx = self.sun_conv1(x)
        xx = self.sun_conv2(xx)
        xx = self.avgpooling(xx)
        xx = xx.reshape(xx.shape[0], -1)
        dense_feature = self.sun_para_dense1(xx)

        if gth_ground_shlight is None:
            sun_para = self.sun_para_dense2(
                torch.cat([dense_feature, ground_shlight], dim=1)
            )
        else:
            sun_para = self.sun_para_dense2(
                torch.cat(
                    [
                        dense_feature,
                        gth_ground_shlight.reshape(gth_ground_shlight.shape[0], -1),
                    ],
                    dim=1,
                )
            )

        sun_para = sun_para.reshape(sun_para.shape[0], 3)
        sun = self.sun_dense(torch.cat([xx, env_conf], dim=1))
        sun = sun.reshape(sun.shape[0], 1, 16, 32)

        ground_shlight = ground_shlight.reshape(ground_shlight.shape[0], 3, 9)

        return ground_shlight, sun, sun_para

    def forward(self, human, mask, gth_ground_shlight=None):

        all_f = self.encode(torch.cat([human, mask], dim=1))

        ground_shlight, sun, sun_para = self.decode(all_f, gth_ground_shlight)

        return ground_shlight, sun, sun_para


class LightNet_Alex(nn.Module):
    def __init__(self):
        super(LightNet_Alex, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4),
            nn.GroupNorm(num_groups=16, num_channels=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2
            ),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.GroupNorm(num_groups=16, num_channels=384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=384),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((12, 8)),
        )

        # 需要针对上一层改变view
        self.layer06 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer07 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer08 = nn.Linear(in_features=4096, out_features=3)

        self.layer16 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer17 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer18 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=512), nn.LogSoftmax(dim=1)
        )

        self.layer26 = nn.Sequential(
            nn.Linear(in_features=12 * 8 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer27 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout()
        )

        self.layer28 = nn.Sequential(nn.Linear(in_features=4096, out_features=27),)

    def forward(self, human, mask):
        x = torch.cat([human, mask], dim=1)
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 12 * 8 * 256)
        sun_para = F.relu(self.layer08(self.layer07(self.layer06(x))))
        sun = self.layer18(self.layer17(self.layer16(x))).reshape(-1, 1, 16, 32)
        ground_shlight = self.layer28(self.layer27(self.layer26(x))).reshape(-1, 3, 9)

        return ground_shlight, sun, sun_para
