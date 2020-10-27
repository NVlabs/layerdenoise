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
# Sample network definition
#
# A scaled-down version of Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
#
# https://groups.csail.mit.edu/graphics/rendernet/
#
###############################################################################

class SampleNet(nn.Module):
	def __init__(self, sequenceHeader, tonemapper, num_samples=8, splat=False, use_sample_info=False, kernel_size=17):

		super(SampleNet, self).__init__() 
		self.use_sample_info = use_sample_info
		self.tonemapper      = tonemapper  
		self.output_channels = 128
		self.embed_channels  = 32 
		self.kernel_size     = kernel_size
		self.num_samples     = int(num_samples)
		self.splat           = splat
		self.resolution      = sequenceHeader.resolution
		frameShape           = sequenceHeader.frameShape
		self.input_channels  = frameShape.color[1] + frameShape.normals_depth[1] + frameShape.albedo[1] + frameShape.specular[1] + frameShape.uvt[1] + frameShape.motionvecs[1]

		# Sample Reducer: Maps from input channels to sample embeddings, uses 1x1 convolutions
		self._sample_reducer = nn.Sequential(
			nn.Conv2d(self.input_channels, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
		)

		# Pixel reducer: Used instead of sample reducer for the per-pixel network, uses 1x1 convolutions
		self._pixel_reducer = nn.Sequential(
			nn.Conv2d(self.input_channels*2, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
			nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
			Activation,
		)

		# Kernel generator: Combines UNet per-pixel output with per-sample or per-pixel embeddings, uses 1x1 convolutions
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
		num_weights = self.kernel_size*self.kernel_size

		frame    = sequenceData.frameData[0]
		radiance = frame.color[:, :, 0:3, ...]
		rgb      = self.tonemapper(frame.color)
		
		xc = torch.cat((rgb, frame.normals_depth, frame.albedo, frame.specular, frame.uvt, frame.motionvecs), dim=2)

		# We transform a 5D tensor [batch, sample, channel, weight, width]
		# into a 4D tensor [batch, embedding, weight, width]

		# loop over samples to create embeddings
		sh = xc.shape
		embedding = torch.cuda.FloatTensor(sh[0], sh[1], self.embed_channels, sh[3], sh[4]).fill_(0)

		if self.use_sample_info:
			# loop over samples to create embeddings
			for i in range(sh[1]):
				embedding[:,i,...] = self._sample_reducer(xc[:,i,...])
			avg_embeddings = embedding.mean(dim=1) # average over embeddings dimension
		else:
			# average per-sample info 
			xc_mean = torch.mean(xc, dim=1)
			xc_variance = torch.var(xc, dim=1, unbiased=False)
			embedding[:,0,...] = self._pixel_reducer(torch.cat((xc_mean,xc_variance), dim=1))
			avg_embeddings = embedding[:,0,...]

		context = self._unet(avg_embeddings)
		ones  = torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(1.0)

		if self.use_sample_info: # work on individual samples
			accum   = torch.cuda.FloatTensor(sh[0], 3, sh[3], sh[4]).fill_(0)
			accum_w = torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(0)
			# create sample weights
			sample_weights = torch.cat(tuple(self._kernel_generator(torch.cat((embedding[:, i, ...], context), dim=1)) for i in range(0, self.num_samples)), dim=1)
			weight_max = torch.max(sample_weights, dim=1, keepdim=True)[0]
			sample_weights = torch.exp(sample_weights - weight_max)

			for i in range(self.num_samples): # loop over samples
				startw  = num_weights*(i)
				endw    = num_weights*(i+1)
				accum += self._kpn(radiance[:, i, ...].contiguous(), sample_weights[:, startw:endw, ...].contiguous())
				accum_w += self._kpn(ones.contiguous(), sample_weights[:, startw:endw, ...].contiguous())
			filtered = accum / (accum_w + EPSILON)

		else: # work on pixel aggregates
			radiance_mean = torch.mean(radiance, dim=1)
			pixel_weights = self._kernel_generator(torch.cat((embedding[:,0,...], context), dim=1))
			weight_max = torch.max(pixel_weights, dim=1, keepdim=True)[0]
			pixel_weights = torch.exp(pixel_weights - weight_max)
			col = self._kpn(radiance_mean.contiguous(), pixel_weights)
			w   = self._kpn(ones.contiguous(), pixel_weights)
			filtered = col/(w+EPSILON)

		return utils.object_from_dict({'color' : filtered})

	def inference(self, sequenceData):
		return self.forward(sequenceData, 0)
