# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

###############################################################################
# Weighted filter for kernel predicting networks
###############################################################################

class WeightedFilterPy(torch.nn.Module):
	def __init__(self, in_channels, kernel_size, bias=True, splat=False):
		super(WeightedFilterPy, self).__init__()
		self.in_channels  = in_channels
		self.out_channels = in_channels
		self.kernel_size  = kernel_size
		self.splat        = splat

		if bias:
			self.bias     = torch.nn.Parameter(torch.Tensor(self.out_channels))

	def forward(self, input, weight):

		HEIGHT = input.shape[2]  # assume input is a tensor with shape NCHW
		WIDTH  = input.shape[3]

		v1, v0 = torch.meshgrid([torch.linspace(-1.0, 1.0, HEIGHT).cuda(), torch.linspace(-1.0, 1.0, WIDTH).cuda()])

		offsetx = 2.0 / (WIDTH - 1)
		offsety = 2.0 / (HEIGHT - 1)

		radius = self.kernel_size // 2
		batch_size = input.shape[0]

		out = torch.zeros_like(input)
		for i in range(-radius, radius + 1):
			for j in range(-radius, radius + 1):
				# compute tap offset
				v0_tap = v0 + j*offsetx
				v1_tap = v1 + i*offsety

				mvs = torch.stack((v0_tap, v1_tap), dim=2)

				# shift image according to offset
				tap_col = torch.nn.functional.grid_sample(input, mvs.expand(batch_size,-1,-1,-1), padding_mode='zeros', align_corners=True)

				# If using "splat" kernels, shift weights along with colors
				if self.splat:
					tap_w = torch.nn.functional.grid_sample(weight, mvs.expand(batch_size,-1,-1,-1), padding_mode='zeros', align_corners=True)
					out = out + tap_col[:, :, ...] * tap_w[:, (radius - i)*self.kernel_size + (radius - j), ...].unsqueeze(1)
				else:
					out = out + tap_col[:, :, ...] * weight[:, (i + radius)*self.kernel_size + (j + radius), ...].unsqueeze(1)
	
		if hasattr(self, 'bias'):
			for oc in range(self.out_channels):
				out[:, oc, ...] = out[:, oc, ...] + self.bias[oc].expand_as(out[:, oc, ...])

		return out
