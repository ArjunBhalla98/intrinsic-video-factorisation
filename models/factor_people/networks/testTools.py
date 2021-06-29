import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch


def F(x):
    A = 0.22
    B = 0.30
    C = 0.10
    D = 0.20
    E = 0.01
    F = 0.30

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F + 1e-12)) - E / F


def Uncharted2ToneMapping(color, adapted_lum):
    WHITE = F(11.2)
    tonemapping_img = F(1.6 * adapted_lum * color) / WHITE
    tonemapping_img = (np.clip(tonemapping_img, 0, float("inf")) + 1e-7) ** (1 / 2.2)
    return tonemapping_img


def Uncharted2ToneMapping_tensor(color, adapted_lum):
    WHITE = F(11.2)
    tonemapping_img = F(1.6 * adapted_lum * color) / WHITE
    tonemapping_img = (torch.clamp(tonemapping_img, min=0.0) + 1e-7) ** (1 / 2.2)
    return tonemapping_img


def tonemapping(input):

    return np.clip(Uncharted2ToneMapping(input, 1.0), 0.0, 1.0)


def tonemapping_tensor(input):

    return torch.clamp(Uncharted2ToneMapping_tensor(input, 1.0), 0.0, 1.0)


def RMSE(x, y, mask=None):
    if len(x.shape) == 2:
        dim = 1
    else:
        dim = 3
    sum = x.shape[0] * x.shape[1] * dim
    if not (mask is None):
        x = mask * x
        y = mask * y
        sum = np.sum(mask) * dim
    t = x - y
    return (np.sum(t ** 2) / (1e-12 + sum)) ** 0.5


# def RMSE_s(x, y, mask=None):
#     if not (mask is None):
#         x = mask * x
#         y = mask * y
#     xx = x.reshape(-1, 1)
#     yy = y.reshape(-1)
#     scale = np.linalg.lstsq(xx, yy, rcond=None)[0][0]
#     return RMSE(scale*x, y, mask)


def RMSE_s(x, y, mask=None):
    if not (mask is None):
        x = mask * x
        y = mask * y
    scale = np.sum(x) / (1e-12 + np.sum(y))
    return RMSE(x, scale * y, mask)


def DSSIM(x, y, mask=None):
    if not (mask is None):
        x = mask * x
        y = mask * y
    max_range = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
    return 0.5 * (1.0 - ssim(x, y, data_range=max_range, multichannel=True))


def calErr(x, y, mask=None):
    return RMSE(x, y, mask), RMSE_s(x, y, mask), DSSIM(x, y, mask)


def calErr_tensor(xx, yy, mask_np=None):

    x = xx[0].permute(1, 2, 0).cpu().detach().numpy()
    y = yy[0].permute(1, 2, 0).cpu().detach().numpy()
    if mask_np is not None:
        mask = mask_np[0].permute(1, 2, 0).cpu().detach().numpy()
    else:
        mask = None

    return RMSE(x, y, mask), RMSE_s(x, y, mask), DSSIM(x, y, mask)


def savefig(img, path):
    save_img = np.squeeze(img)
    Image.fromarray((np.clip(save_img * 255, 0.0, 255.0)).astype(np.uint8)).save(path)


def savetensor(tensor, path):
    img = tensor[0].permute(1, 2, 0).cpu().detach().numpy()
    savefig(img, path)


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
            normal[i, j] = normal[i, j] / (1e-12 + np.linalg.norm(normal[i, j]))

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


env_transport_map = getEnvTransport()


def recoverySH(shlight):

    return np.clip(getShading(env_transport_map, shlight), 0.0, float("inf"))


def recoverySH_Tensor(shlight_tensor):
    envlight = torch.Tensor(shlight_tensor.shape[0], 3, 16, 32)
    for i in range(shlight_tensor.shape[0]):
        shlight = shlight_tensor[i].permute(1, 0).cpu().detach().numpy()
        sh_img = recoverySH(shlight)
        envlight[i] = torch.FloatTensor(sh_img.transpose(2, 0, 1))
    return envlight.cuda()


def recoveryEnvLight(ground, sun_map_exp, sun_intensity):

    sun_map_exp = sun_map_exp / torch.max(
        sun_map_exp.reshape(sun_map_exp.shape[0], -1), dim=1
    )[0].reshape(sun_map_exp.shape[0], 1, 1, 1)
    ground_map = recoverySH_Tensor(ground)
    light_full = ground_map + sun_map_exp * sun_intensity.reshape(
        sun_intensity.shape[0], 3, 1, 1
    )

    return light_full


def ResizeImage(img_raw, mask_raw, scene=None):

    img_shape = img_raw.shape
    sx, sy = img_raw.shape[0] // 2, img_raw.shape[1] // 2

    print(img_raw.shape, mask_raw.shape)
    mask = np.zeros((img_shape[0] * 2, img_shape[1] * 2), dtype=np.uint8)
    mask[sx : sx + img_shape[0], sy : sy + img_shape[1]] = mask_raw

    img = np.zeros((img_shape[0] * 2, img_shape[1] * 2, 3), dtype=np.uint8)
    img[sx : sx + img_shape[0], sy : sy + img_shape[1]] = img_raw

    for i in range(sx):
        img[i, sy : sy + img_shape[1]] = img_raw[i % 5, :]
    for i in range(sx + img_shape[0], img_shape[0] * 2):
        img[i, sy : sy + img_shape[1]] = img_raw[img_shape[0] - i % 5 - 1, :]

    for i in range(sy):
        img[:, i] = img[:, sy + i % 5]
    for i in range(sy + img_shape[1], img_shape[1] * 2):
        img[:, i] = img[:, sy + img_shape[1] - 1 - i % 5]

    x_trimmed, y_trimmed, w_trimmed, h_trimmed = cv2.boundingRect(mask)

    mask_length = 715 - 214
    top_padding = int((214 / mask_length) * h_trimmed)

    scale = h_trimmed / mask_length

    middle_x = x_trimmed + w_trimmed // 2

    top_x = middle_x - int(scale * 256)
    top_y = y_trimmed - top_padding

    width = int(scale * 512)
    height = int(scale * 768)

    botton_x = top_x + width
    botton_y = top_y + height

    tmp_img = img[top_y:botton_y, top_x:botton_x]
    tmp_mask = mask[top_y:botton_y, top_x:botton_x]

    trimmed_img = cv2.resize(tmp_img, (512, 768), interpolation=cv2.INTER_AREA)
    trimmed_mask = cv2.resize(tmp_mask, (512, 768), interpolation=cv2.INTER_NEAREST)

    if scene is not None:
        if not (scene.shape[0] == 768 and scene.shape[1] == 512):
            if scene.shape[1] * 1.5 > scene.shape[0]:
                padding = int((scene.shape[1] - scene.shape[0] / 1.5) / 2)
                scene = cv2.resize(
                    scene[:, padding : scene.shape[1] - padding],
                    (512, 768),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                scene = cv2.resize(
                    scene[scene.shape[0] - int(scene.shape[1] * 1.5) :, :],
                    (512, 768),
                    interpolation=cv2.INTER_AREA,
                )

    return trimmed_img, trimmed_mask, scene


def GetSceneList(Scene_img, human_num, offset=0):

    scale = 768 / Scene_img.shape[0]
    resized_width = int(scale * Scene_img.shape[1])
    resized_scene = cv2.resize(
        Scene_img, (resized_width, 768), interpolation=cv2.INTER_AREA
    )

    interval = int((resized_width - 512 - offset * 2) / (human_num - 1))
    offset_begin = (resized_width - 512 - interval * (human_num - 1)) // 2
    offset_end = resized_width - 512 - offset_begin - interval * (human_num - 1)

    img_list = []
    for i in range(human_num):
        img = np.zeros((768, 512, 3), dtype=np.uint8)
        img[:, :, :] = resized_scene[
            :, offset_begin + i * interval : offset_begin + i * interval + 512, :
        ]
        img_list.append(img)

    return resized_scene, img_list, offset_begin, offset_end, interval


def CompositeScene(scene, img_list, interval, offset_begin, offset_end):
    target = np.zeros_like(scene)
    width = target.shape[1]
    target[:, 0:offset_begin] = scene[:, 0:offset_begin]
    target[:, -offset_end:] = scene[:, -offset_end:]

    img_width = 512

    end_length = (img_width + interval) // 2
    begin_length = end_length - interval

    target[:, offset_begin : offset_begin + begin_length] = img_list[0][
        :, 0:begin_length
    ]
    target[
        :, width - offset_end - (img_width - end_length) : width - offset_end
    ] = img_list[-1][:, -(img_width - end_length) :]

    for i, img in enumerate(img_list):
        target[
            :,
            offset_begin
            + i * interval
            + begin_length : offset_begin
            + i * interval
            + end_length,
        ] = img[:, begin_length:end_length]

    return target


def GetBg(Scene_img):

    scale = 768 / Scene_img.shape[0]
    resized_width = int(scale * Scene_img.shape[1])
    resized_scene = cv2.resize(
        Scene_img, (resized_width, 768), interpolation=cv2.INTER_AREA
    )

    bg = resized_scene[:, -512:, :]

    return resized_scene, bg


def CompositeBg(resized_scene, human):

    target = resized_scene.copy()
    target[:, -512:, :] = human
    return target


def composition(albedo, shading, mask, bg, shadow):
    return (
        tonemapping_tensor(albedo * shading) * mask + shadow * bg * (1.0 - mask)
    ).contiguous()


def composition_noshadow(albedo, shading, mask, bg):
    return (
        tonemapping_tensor(albedo * shading) * mask + bg * (1.0 - mask)
    ).contiguous()


def composition_wo_tonemapping(albedo, shading, mask, bg, shadow):
    return (
        tonemapping_tensor(albedo) * shading * mask + shadow * bg * (1.0 - mask)
    ).contiguous()


class ImageCompositor:
    def __init__(self, row_num=2, col_num=5, img_shape=(768, 512)):
        self.img_shape = img_shape
        self.row_num = row_num
        self.col_num = col_num

        self.row_interval = img_shape[0] // 8
        self.col_interval = img_shape[1] // 8

        self.img = np.zeros(
            (
                row_num * (img_shape[0] + self.row_interval) + self.row_interval,
                col_num * (img_shape[1] + self.col_interval) + self.col_interval,
                3,
            ),
            np.uint8,
        )

    def put_img(self, img, r, c):
        assert img.shape[0:2] == self.img_shape
        assert img.dtype == np.uint8

        st_r = r * (self.img_shape[0] + self.row_interval) + self.row_interval
        st_c = c * (self.img_shape[1] + self.col_interval) + self.col_interval

        self.img[st_r : st_r + self.img_shape[0], st_c : st_c + self.img_shape[1]] = img

    def put_normalized_img(self, normalized_img, r, c):
        img = (np.clip(normalized_img * 255, 0, 255)).astype(np.uint8)
        self.put_img(img, r, c)

    def put_tensor(self, tensor, r, c):
        img = tensor[0].permute(1, 2, 0).cpu().detach().numpy()
        img = (np.clip(img * 255, 0, 255)).astype(np.uint8)

        self.put_img(img, r, c)

    def save(self, path):
        Image.fromarray(self.img).save(path)
