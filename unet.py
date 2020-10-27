# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn

import utils

###############################################################################
# Activation
###############################################################################

Activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

###############################################################################
# Regular U-net
###############################################################################

class UNet(nn.Module):
	def __init__(self, input_channels, output_channels, 
		encoder_features=[[64, 64], [128], [256], [512], [512]], 
		bottleneck_features=[512], 
		decoder_features=[[512, 512], [256, 256], [128, 128], [64, 64], [64, 64]]):
		super().__init__() 
		
		self.output_channels     = output_channels
		self.input_channels      = input_channels
		self.encoder_features    = encoder_features
		self.bottleneck_features = bottleneck_features
		self.decoder_features    = decoder_features
		self.initNetwork()

	def initNetwork(self):
		# Utility function that creates a convolution "block" from a list of features, 
		# with one convolutional layer per feature count in the list
		def make_conv_block(in_features, features):
			layers = []
			prev_features = in_features
			for f in features:
				layers = layers + [nn.Conv2d(prev_features, f, 3, padding=1), Activation]
				prev_features = f
			return layers

		prev_features = self.input_channels

		# Create encoder
		enc = []
		for enc_f in self.encoder_features:
			enc = enc + [nn.Sequential(*make_conv_block(prev_features, enc_f), nn.MaxPool2d(2))]
			prev_features = enc_f[-1]
		self.enc = nn.ModuleList(enc)

		# Create bottleneck
		self.bottleneck = nn.Sequential(*make_conv_block(prev_features, self.bottleneck_features)).cuda()
		prev_features = self.bottleneck_features[-1]

		# Create decoder
		dec = []
		for idx, dec_f in enumerate(self.decoder_features[:-1]):
			skip_features = self.encoder_features[len(self.decoder_features) - idx - 2][-1]
			dec = dec + [nn.Sequential(*make_conv_block(prev_features + skip_features, dec_f)).cuda()]
			prev_features = dec_f[-1]
		dec = dec + [nn.Sequential(*make_conv_block(prev_features + self.input_channels, self.decoder_features[-1])).cuda()]
		self.dec = nn.ModuleList(dec)

		# Final layer
		self.final = nn.Conv2d(self.decoder_features[-1][-1], self.output_channels, 3, padding=1)

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, prev):

		# Run encoder
		enc_vars = [prev]
		for block in self.enc:
			prev = block(prev)
			enc_vars = enc_vars + [prev]

		# Run bottleneck
		prev = self.bottleneck(prev)
		
		# Run decoder
		for idx, block in enumerate(self.dec):
			prev = nn.functional.interpolate(prev, scale_factor=2, mode='nearest', align_corners=None) # Upscale result from previous step
			concat = torch.cat((prev, enc_vars[len(self.dec) - idx - 1]), dim=1)                       # Concatenate skip connection
			prev = block(concat)

		# Run final composition
		output = self.final(prev)

		# Return output color & all decoder levels
		return output


