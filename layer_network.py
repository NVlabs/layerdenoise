# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch_utils import WeightedFilter

from unet import *
from utils import *

EPSILON = 0.000001 # Small epsilon to avoid division by zero

###############################################################################
# Layer network definition
#
# The large network described in "Neural Denoising with Layer Embeddings"
#
# https://research.nvidia.com/publication/2020-06_Neural-Denoising-with
#
###############################################################################

class LayerNet(nn.Module):
	def __init__(self, sequenceHeader, tonemapper, splat, num_samples, kernel_size):

		super(LayerNet, self).__init__() 
		self.tonemapper      = tonemapper  
		self.output_channels = 128
		self.embed_channels  = 32 
		self.kernel_size     = kernel_size
		self.num_samples     = int(num_samples)
		self.splat           = splat
		self.resolution      = sequenceHeader.resolution
		frameShape           = sequenceHeader.frameShape
		self.input_channels  = frameShape.color[1] + frameShape.normals_depth[1] + frameShape.albedo[1] + frameShape.specular[1] + frameShape.uvt[1] + frameShape.motionvecs[1]
		self.layers          = 2

		# Sample reducer: Maps from input channels to sample embeddings, uses 1x1 convolutions
		self._red1 = nn.Sequential(
			nn.Conv2d(self.input_channels, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
		)

		# Sample partitioner: Computes weights for splatting samples to layers, uses 1x1 convolutions
		self._sample_partitioner = nn.Sequential(
			nn.Conv2d(self.output_channels+self.embed_channels, 32, 1, padding=0),
			Activation,
			nn.Conv2d(32, 16, 1, padding=0),
			Activation,
			nn.Conv2d(16, self.layers, 1, padding=0), # One splat weight per layer
		)

		# Kernel generator: Computes filter kernels per-layer, uses 1x1 convolutions
		self._kernel_generator = nn.Sequential(
			nn.Conv2d(self.output_channels+self.embed_channels, 128, 1, padding=0),
			Activation,
			nn.Conv2d(128, 128, 1, padding=0),
			Activation,
			nn.Conv2d(128, self.kernel_size*self.kernel_size, 1, padding=0), # output kernel weights
		)

		# U-Net: Generates context features
		self._unet = UNet(self.embed_channels, self.output_channels, encoder_features=[[64, 64], [128], [256], [512], [512]], bottleneck_features=[512], decoder_features=[[512, 512], [256, 256], [128, 128], [128, 128], [128, 128]])

		# Filter for applying predicted kernels
		self._kpn = WeightedFilter(channels=3, kernel_size=self.kernel_size, bias=False, splat=self.splat)

		# Initialize network weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, sequenceData, epoch):
		frame = sequenceData.frameData[0]

		radiance = frame.color # Linear color values
		
		rgb           = self.tonemapper(frame.color)
		normals_depth = frame.normals_depth
		motionvecs    = frame.motionvecs
		albedo        = frame.albedo
		specular      = frame.specular
		uvt           = frame.uvt

		xc = torch.cat((rgb, normals_depth, motionvecs, albedo, specular, uvt), dim=2)

		# loop over samples to create embeddings
		sh = xc.shape
		embedding = torch.cuda.FloatTensor(sh[0], sh[1], self.embed_channels, sh[3], sh[4]).fill_(0)
		for i in range(sh[1]):
			embedding[:, i, ...] = self._red1(xc[:,i,...])
		avg_embeddings = embedding.mean(dim=1) # average over embeddings dimension

		# Run U-net
		context = self._unet(avg_embeddings)

		# Allocate buffers
		l_radiance = [torch.cuda.FloatTensor(sh[0], 3, sh[3], sh[4]).fill_(0) for i in range(self.layers)]
		l_weights = [torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(0) for i in range(self.layers)]		
		l_n = [torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(0) for i in range(self.layers)]
		l_e = [torch.cuda.FloatTensor(sh[0], self.embed_channels, sh[3], sh[4]).fill_(0) for i in range(self.layers)]

		# Splat samples to layers
		for i in range(0, self.num_samples): # loop over samples
			w = self._sample_partitioner(torch.cat((embedding[:, i, ...], context), dim=1))
			w = torch.softmax(w, dim=1) / self.num_samples

			for j in range(self.layers):
				l_radiance[j] += radiance[:, i, ...] * w[:, j:j+1, ...]
				l_weights[j] += w[:, j:j+1, ...]
				l_e[j] += embedding[:, i, ...] * w[:, j:j+1, ...]
				l_n[j] += torch.sum(w[:, j:self.layers, ...], dim=1, keepdim=True) # increment only for samples in or in front

		# Generate layer weights and take exp to make them positive
		layer_weights = torch.cat(tuple(self._kernel_generator(torch.cat((l_e[i], context), dim=1)) for i in range(self.layers)), dim=1)
		weight_max = torch.max(layer_weights, dim=1, keepdim=True)[0]
		layer_weights = torch.exp(layer_weights - weight_max) # subtract largest weight for stability
		num_weights = self.kernel_size*self.kernel_size

		# Alpha-blending compositing
		col_sum   = torch.cuda.FloatTensor(sh[0], 3, sh[3], sh[4]).fill_(0)
		k = torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(1.0)
		for j in range(self.layers):
			startw   = num_weights*j
			endw     = num_weights*(j+1)
			kernel   = layer_weights[:, startw:endw, ...]

			filtered_rad = self._kpn(l_radiance[j].contiguous(), kernel.contiguous())
			alpha        = self._kpn(l_weights[j].contiguous(), kernel.contiguous())
			filtered_n   = self._kpn(l_n[j].contiguous(), kernel.contiguous())
			filtered_rad = filtered_rad / (filtered_n + EPSILON)
			alpha        = alpha / (filtered_n + EPSILON)
			col_sum     += filtered_rad * k
			k            = (1.0 - alpha) * k

		return utils.object_from_dict({'color' : col_sum})

	def inference(self, sequenceData):
		return self.forward(sequenceData, 0)
