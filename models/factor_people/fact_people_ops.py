import torch
import numpy as np
from PIL import Image
import torch.nn as nn

import models.factor_people.networks.network_light as network_light
import models.factor_people.networks.network as network
import models.factor_people.networks.BilateralLayer as BlLayer
import models.factor_people.networks.testTools as testTools


# define a class that takes input image and return factors and a reconstruction or relighting, or insertion
class FactorsPeople:
    # load models
    def __init__(self, all_dirs, device=torch.device("cuda")):
        super(FactorsPeople, self).__init__()

        self_shading_net_path = all_dirs["self_shading_net"]
        shading_net_path = all_dirs["shading_net"]
        SH_model_path = all_dirs["SH_model"]
        albedo_net_path = all_dirs["albedo_net"]
        shadow_net_path = all_dirs["shadow_net"]
        refine_rendering_net_path = all_dirs["refine_rendering_net"]

        # load models
        self.albedo_net = network.Unet_Blurpooling_General(input_channel=7)
        self.albedo_net = nn.DataParallel(self.albedo_net)
        checkpoint = torch.load(albedo_net_path)
        self.albedo_net.module.load_state_dict(checkpoint["model"])
        self.albedo_net.cuda_kernels()

        self.SH_model = network_light.LightNet_Hybrid(16, input_channel=4)
        self.SH_model = nn.DataParallel(self.SH_model)
        checkpoint = torch.load(SH_model_path)
        self.SH_model.module.load_state_dict(checkpoint["model"])

        self.shading_net = network.Unet_Blurpooling_General_Light()
        self.shading_net = nn.DataParallel(self.shading_net)
        checkpoint = torch.load(shading_net_path)
        self.shading_net.module.load_state_dict(checkpoint["model"])
        self.shading_net.cuda_kernels()

        self.self_shading_net = network.SepNetComplete_Shading(f_channel=16)
        self.self_shading_net = nn.DataParallel(self.self_shading_net)
        checkpoint = torch.load(self_shading_net_path)
        self.self_shading_net.module.load_state_dict(checkpoint["model"])

        self.shadow_net = network.Unet_Blurpooling_Shadow()
        self.shadow_net = nn.DataParallel(self.shadow_net)
        checkpoint = torch.load(shadow_net_path)
        self.shadow_net.module.load_state_dict(checkpoint["model"])
        self.shadow_net.cuda_kernels()

        self.refine_rendering_net = network.Unet_Blurpooling_General_Light(
            input_channel=6
        )
        self.refine_rendering_net = nn.DataParallel(self.refine_rendering_net)
        checkpoint = torch.load(refine_rendering_net_path)
        self.refine_rendering_net.module.load_state_dict(checkpoint["model"])
        self.refine_rendering_net.cuda_kernels()

        self.refine_net = BlLayer.BilateralSolver()

    def set_eval(self):
        self.albedo_net.eval()
        self.SH_model.eval()
        self.shading_net.eval()
        self.self_shading_net.eval()
        self.shadow_net.eval()
        self.refine_rendering_net.eval()

    def load_model_state(self, model_state_dict):
        self.self_shading_net.module.load_state_dict(
            torch.load(model_state_dict["self_shading_net"])
        )
        self.shading_net.module.load_state_dict(
            torch.load(model_state_dict["shading_net"])
        )
        self.SH_model.module.load_state_dict(torch.load(model_state_dict["SH_model"]))
        self.albedo_net.module.load_state_dict(
            torch.load(model_state_dict["albedo_net"])
        )
        self.shadow_net.module.load_state_dict(
            torch.load(model_state_dict["shadow_net"])
        )
        self.refine_rendering_net.module.load_state_dict(
            torch.load(model_state_dict["refine_rendering_net"])
        )

    def get_image(self, img_path, mask_path):
        input_img_origin = np.array(Image.open(img_path).resize((278, 500)))
        input_mask_origin = np.array(Image.open(mask_path).resize((278, 500)))
        if input_mask_origin.shape[0:2] != input_img_origin.shape[0:2]:
            input_img_origin = input_img_origin.transpose(1, 0, 2)
            input_img_origin = input_img_origin[:, ::-1, :]
        if len(input_mask_origin.shape) > 2:
            input_mask_origin = input_mask_origin[:, :, 0]

        input_img, input_mask, _ = testTools.ResizeImage(
            input_img_origin, input_mask_origin, None
        )
        input_mask = np.expand_dims(input_mask, axis=2)

        img_shape = (768, 512)
        human_mask = torch.Tensor(1, 1, img_shape[0], img_shape[1])
        human_input = torch.Tensor(1, 3, img_shape[0], img_shape[1])

        input_img = input_img / 255.0
        input_mask = input_mask / 255.0

        human_mask[0] = torch.FloatTensor(input_mask.transpose(2, 0, 1))
        human_input[0] = torch.FloatTensor(input_img.transpose(2, 0, 1))

        mask = human_mask
        tonemapped = human_input

        return tonemapped, mask

    def get_lighting(self, image, mask, raw_output=False):
        (
            est_ground,
            est_sun_map,
            est_sun_intensity,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.SH_model(image, mask, None, None)

        est_light = testTools.recoveryEnvLight(
            est_ground, torch.exp(est_sun_map), est_sun_intensity
        )

        if raw_output:
            return {
                "ground": est_ground,
                "sun_map": est_sun_map,
                "sun_intensity": est_sun_intensity,
            }
        else:
            return est_light

    def get_shading(self, image, mask, light=None, isSelf=True):
        if light is None:
            light = self.get_lighting(image, mask)
        if isSelf:
            est_shading = self.self_shading_net(image, mask, light)
        else:
            est_shading = self.shading_net(torch.cat([image, mask], dim=1), light)
        est_shading = est_shading * mask
        return est_shading

    def get_albedo(self, image, mask, shading=None):
        if shading is None:
            shading = self.get_shading(image, mask)
        est_albedo = self.albedo_net(torch.cat([image, shading, mask], dim=1))
        est_albedo = torch.clamp(est_albedo, 0.0, 1.0)
        est_albedo = est_albedo * mask

        # shading_map = torch.sum(shading, dim=1, keepdim=True)  # ???
        # shading_map = shading_map / torch.clamp(shading_map.max(), min=1e-5)
        # est_albedo = (
        #     self.refine_net.apply(est_albedo, est_albedo, shading_map) * mask
        # )  # **** ??
        return est_albedo

    def get_shadow(self, image, mask, light=None):
        if light is None:
            light = self.get_lighting(image, mask)
        est_shadow = self.shadow_net(torch.cat([image, mask], dim=1), light)
        return est_shadow

    def factor(self, image, mask):
        est_light = self.get_lighting(image, mask)
        est_shading = self.get_shading(image, mask, est_light)
        est_albedo = self.get_albedo(image, mask, est_shading)
        est_shadow = self.get_shadow(image, mask, est_light)
        return {
            "light": est_light,
            "shading": est_shading,
            "albedo": est_albedo,
            "shadow": est_shadow,
        }

    def reconstruct(self, image, mask):
        factors = self.factor(image, mask)
        reconstruct = testTools.composition(
            factors["albedo"], factors["shading"], mask, image, factors["shadow"]
        )
        return reconstruct, factors

    def relight_withpeople(
        self, source_image, source_mask, target_image, target_mask, target_background
    ):
        est_source_albedo = self.get_albedo(source_image, source_mask)

        est_target_light = self.get_lighting(target_image, target_mask)
        est_target_shading = self.get_shading(
            source_image, source_mask, est_target_light
        )
        est_target_shadow = self.get_shadow(source_image, source_mask, est_target_light)

        relighted = testTools.composition(
            est_source_albedo,
            est_target_shading,
            source_mask,
            target_background,
            est_target_shadow,
        )
        return (
            relighted,
            est_source_albedo,
            est_target_light,
            est_target_shading,
            est_target_shadow,
        )

    def relight_emptyscene(self, source_image, source_mask, target_bg, target_light):
        est_source_albedo = self.get_albedo(source_image, source_mask)

        est_target_shading = self.get_shading(
            source_image, source_mask, target_light, isSelf=False
        )
        est_target_shadow = self.get_shadow(source_image, source_mask, target_light)

        relighted = testTools.composition(
            est_source_albedo,
            est_target_shading,
            source_mask,
            target_bg,
            est_target_shadow,
        )
        return relighted, est_source_albedo, est_target_shading, est_target_shadow

    def refine(self, original, relighted, light, mask, bg, est_shadow):
        refined = self.refine_rendering_net(
            torch.cat([relighted, original], dim=1), light
        )
        refined = torch.clamp(refined, 0.0, 1.0)

        refined = refined * mask + bg * est_shadow * (1 - mask)
        return refined


def get_model_dirs():
    """
    Just returns the dict with paths on the server where the larger models are kept.
    Important: It is easiest to initialise this class with this dict, then
    use factorpeople.load_model_state() with the new pth files
    """
    model_dir = "/phoenix/S7/js2625/SIGGRAPH_InsertHuman/desktopmini/"
    all_dirs = {
        "self_shading_net": model_dir
        + "models/self_shading.pth",  # /data/human-inserting/logs/sepnet_shading/sepnet_f16_lr4e-5_dssimonly'
        "shading_net": model_dir
        + "models/shading.pth",  # /data/human-inserting/logs/sepnet_shading/unetblur_lr4e-5_dssimonly_noself'
        "SH_model": model_dir
        + "models/SH.pth",  # /data/human-inserting/logs/sepnet_light_sh_single/hybrid_position0_map0.5'
        "albedo_net": model_dir
        + "models/albedo.pth",  # /data/human-inserting/logs/sepnet_albedo/unetblur_l1_perp_w_shading'
        "shadow_net": model_dir
        + "models/shadow.pth",  # /data/human-inserting/logs/sepnet_others/shadow_unetblur_l1loss'
        "refine_rendering_net": model_dir
        + "models/refine.pth",  # /data/human-inserting/logs/sepnet_others/unetblur_relighting_refine_l2_perp'
    }

    return all_dirs


if __name__ == "__main__":
    img_file = "/phoenix/S3/ab2383/data/train_imgs/00110_0015.png"
    mask_file = "/phoenix/S3/ab2383/data/train_imgs/00110_0015_mask.png"
    all_dirs = get_model_dirs()

    factorspeople = FactorsPeople(all_dirs)

    img, mask = factorspeople.get_image(img_file, mask_file)
    with torch.no_grad():
        pred_factors = factorspeople.factor(img.cuda(), mask.cuda())
    print(pred_factors.keys())
    print(pred_factors["light"].shape)
    print(pred_factors["shading"].shape)
    print(pred_factors["albedo"].shape)
    print(pred_factors["shadow"].shape)
