import EfficientUnet_PyTorch
from EfficientUnet_PyTorch.efficientunet.layers import *
from EfficientUnet_PyTorch.efficientunet.efficientnet import EfficientNet

import torch.nn.functional as F
from collections import OrderedDict

import torchvision.ops.roi_align as roi_align

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']

def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True, bottleneck_inject_dims=0):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels+bottleneck_inject_dims, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        else:
            self.up_conv5 = up_conv(64, 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
                           
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x=None, bottleneck_inject_x=None, encoder_fts=None):

        if encoder_fts is None:
            input_ = x
            blocks = get_blocks_to_be_concat(self.encoder, x)
            _, ft = blocks.popitem() # 24x16
        else:
            ft = encoder_fts[0] # 24x16

        # concat with additional bottle neck input, if available
        if bottleneck_inject_x is not None:
            inject_x_adjusted = F.interpolate(bottleneck_inject_x, 
                                                size=(ft.shape[2],ft.shape[3]), 
                                                mode='bilinear', 
                                                align_corners=False)
            x = torch.cat((ft, inject_x_adjusted), dim=1)
            x = self.up_conv1(x)
        else:
            x = self.up_conv1(ft) # ft: 24x16, x: 48x32
        
        if encoder_fts is None:
            x1 = blocks.popitem()[1]
        else:
            x1 = encoder_fts[1] # 48x32
        x = torch.cat([x, x1], dim=1) # check shape
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        if encoder_fts is None:
            x2 = blocks.popitem()[1]
        else:
            x2 = encoder_fts[2]
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        if encoder_fts is None:
            x3 = blocks.popitem()[1]
        else:
            x3 = encoder_fts[3]
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        if encoder_fts is None:
            x4 = blocks.popitem()[1]
        else:
            x4 = encoder_fts[4]
        x = torch.cat([x, x4], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)
        else:
            x = self.up_conv5(x)

        x = self.final_conv(x)

        return {
            'res':x, 
            'encoder_fts': [ft, x1, x2, x3, x4]
        }

class DenseLightNet(nn.Module):
    def __init__(self, light_dims=128, efficientnet_version='b3'):
        super(DenseLightNet, self).__init__()
        encoder = EfficientNet.encoder('efficientnet-{}'.format(efficientnet_version), pretrained=True)
        self.lightnet = EfficientUnet(encoder, out_channels=light_dims)

        self.roi_output_size = [4,3]

    def forward(self, image, mask_boxes=None):
        # get per-pixel light map
        denselight = self.lightnet(image)['res']
        # get pooled light features, if maskes are given
        if mask_boxes is not None:
            roilight = roi_align(denselight, mask_boxes, output_size=self.roi_output_size)
            return roilight
        else:
            return denselight

if __name__ == "__main__":
    model = DenseLightNet(light_dims=128)
    model.cuda()

    # no mask
    image = torch.randn(2,3,256,256).cuda()
    res = model(image)
    print(res.shape)

    # with mask
    mask = [torch.Tensor([[100,100,200,200]]).cuda(), 
            torch.Tensor([[110,110,300,300]]).cuda()]
    res = model(image, mask)
    print(res.shape)