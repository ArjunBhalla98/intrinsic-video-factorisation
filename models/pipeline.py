import torch.nn as nn
from .denselight_model import DenseLightNet
from .decoder import Decoder


class DLNetPipeline(nn.Module):
    def __init__(self):
        self.denselight_net = DenseLightNet()
        self.person_detector = (
            None  # Fill in with model that gets people bounding boxes
        )
        self.albedo_net = None  # Fill in with albedo net - google or our FT?
        self.decoder = Decoder()

    def forward(self, frame):
        # Get people prediction for albedos
        figure_boxes = self.person_detector(frame)

	# Denselight and ROI pooling
        lightmap_prediction = self.denselight_net(frame)

	# Human albedos - possibly separate them first
        albedos = self.albedo_net(frame, figure_boxes)

        reconstruction = self.decoder(albedos, lightmap_prediction)
	
	return reconstruction

